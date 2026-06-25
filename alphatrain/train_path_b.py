"""AlphaTrain V-corpus distillation with optional target sharpening.

Trains a policy network on MCTS visit-distribution targets (V11/V12/V13/...).
Color permutation + dihedral augmentation are on by default (color was the
+4% lift in Path B v1's ablation; HISTORY 143-144). Target sharpening via
`--target-temperature` is the load-bearing knob that produced sharp_50's
+57% gameplay lift over baseline (HISTORY 153-156).

The legacy name "train_path_b" comes from the Path B v1 oracle experiment
(soft-KL auxiliary loss). That branch is closed (HISTORY 145, 151-152).
This script is now V-corpus distillation only — no oracle path, no λ tuning,
no auxiliary corpus. The training recipe that produces pillar3a / pillar3b /
pillar3c / ... is just this.

Usage (Colab, ~12h for 17 epochs on G4 / L4):
    python -m alphatrain.train_path_b \\
        --tensor-file alphatrain/data/v13_pillar3a.pt \\
        --amp --compile \\
        --resume alphatrain/data/pillar3a.pt --warm-start \\
        --epochs 17 --batch-size 32768 --lr 3e-4 --warmup-epochs 1 \\
        --target-temperature 0.5 \\
        --copy-to /content/drive/MyDrive/alphatrain/pillar3b_best.pt \\
        --save-dir /content/checkpoints/pillar3b

To disable color augmentation (e.g., for ablation), pass --no-color-augment.
To disable dihedral augmentation, pass --no-dihedral-augment. Default is
both on.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alphatrain.counterfactual import (
    build_corpus as build_aux_corpus,
    build_crisis_corpus,
    listwise_margin_loss,
    soft_correction_loss,
    preflight_metrics as aux_preflight_metrics,
    DEFAULT_MARGIN as AUX_DEFAULT_MARGIN,
    DEFAULT_TOP1_WEIGHT as AUX_DEFAULT_TOP1_WEIGHT,
    DEFAULT_OTHER_WEIGHT as AUX_DEFAULT_OTHER_WEIGHT,
)
from alphatrain.dataset import NUM_CHANNELS, TensorDatasetGPU
from alphatrain.model import AlphaTrainNet, count_parameters


import contextlib


@contextlib.contextmanager
def frozen_bn(model):
    """Put BatchNorm layers in eval mode for the duration of a forward pass.

    The aux corpus is out-of-distribution (dense late-game crisis boards). A
    train-mode aux forward UPDATES BN running mean/var toward that distribution
    — and inference uses those running stats, so the deployed policy gets
    poisoned (train loss stays fine on batch stats while val/gameplay degrade;
    see the BN-contamination lessons). Splitting the forward fixed the loss
    contamination but NOT the running-stat update. Freezing BN here makes the
    aux forward normalize with the SAME running stats inference uses (so the
    aux teaches inference-mode behavior) and stops the poisoning. Affine
    params + conv/linear weights still receive gradients.
    """
    bns = [m for m in model.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    prev = [m.training for m in bns]
    for m in bns:
        m.eval()
    try:
        yield
    finally:
        for m, p in zip(bns, prev):
            m.train(p)


def cross_entropy_soft(logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def distillation_loss(logits, soft_targets, blend_alpha=1.0,
                      target_temperature=1.0, decisiveness_power=0.0):
    """Cross-entropy on a visit-distribution target.

    target_temperature=1.0 (default): targets used as-stored.
    target_temperature<1.0: sharpen targets via `target**(1/T)` renormalized.
        Forces the model to commit to high-visit moves. Engaged sharp_50's
        +57% lift over baseline.
    blend_alpha<1.0: convex blend of soft target CE and hard CE on argmax.
        Rarely used; legacy from pre-sharpening experiments.
    decisiveness_power>0: weight each state's CE by its RAW visit peakedness
        (top-share)**power, normalized to mean 1. Decisive escape-or-die states
        (top-share ~0.6-0.9) dominate the gradient; flat quiet states (top-share
        ~0.2, "many moves okay") get near-zero weight so they CANNOT flatten the
        policy. This separates the floor signal from the flat quiet bulk that
        de-peaks a strong base under uniform full-corpus distillation.
    """
    w = None
    if decisiveness_power > 0.0:
        # peakedness from the RAW (pre-sharpen) target = the position's decisiveness.
        raw_top = soft_targets.max(dim=-1).values.clamp(min=1e-6)   # (B,)
        w = raw_top.pow(decisiveness_power)
        w = w / w.mean().clamp(min=1e-8)                            # mean -> 1
    if target_temperature != 1.0:
        sharp = soft_targets.pow(1.0 / target_temperature)
        sharp = sharp / sharp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        soft_targets = sharp
    log_probs = F.log_softmax(logits, dim=-1)
    per = -(soft_targets * log_probs).sum(dim=-1)                   # (B,)
    soft = (w * per).mean() if w is not None else per.mean()
    if blend_alpha >= 1.0:
        return soft
    argmax_idx = soft_targets.argmax(dim=-1)
    hard_per = F.cross_entropy(logits, argmax_idx, reduction='none')
    hard = (w * hard_per).mean() if w is not None else hard_per.mean()
    return blend_alpha * soft + (1.0 - blend_alpha) * hard


def _aux_lambda_schedule(step_in_epoch, steps_per_epoch, epoch,
                          target_lambda, warmup_epochs):
    """Linear ramp 0 → target_lambda over `warmup_epochs` epochs."""
    if warmup_epochs <= 0:
        return target_lambda
    global_step = epoch * steps_per_epoch + step_in_epoch
    warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))
    return target_lambda * min(1.0, global_step / warmup_steps)


_GRAD_AUDIT = []   # accumulates (|g_main|, |g_aux|, lam, cos) for --grad-audit


def train_epoch(model, loader, optimizer, device, scaler, amp_dtype,
                 log_interval=100, blend_alpha=1.0, target_temperature=1.0,
                 aux=None, epoch=0, grad_audit=0, decisiveness_power=0.0):
    """One epoch. Optionally adds the listwise margin aux loss.

    `aux`, when not None, is a dict with:
        obs, winner_idx, top1_idx, loser_idx, loser_mask  — all on device
        batch_size, target_lambda, margin, top1_weight, other_weight,
        warmup_epochs, preflight_every, abort_flip_rate (optional float)
        ptr: persistent int across batches (passed as a mutable list).
    """
    model.train(True)
    # Aux forward goes through the UNCOMPILED module (shares parameters with the
    # compiled `model`). Toggling BN.eval() on a torch.compile'd model is
    # unreliable — the compiled graph may have specialized on training=True, so
    # the frozen-BN switch might not take effect. The eager module respects it.
    base_model = getattr(model, '_orig_mod', model)
    use_amp = amp_dtype != torch.float32
    total_loss = 0.0
    total_aux = 0.0
    aux_steps = 0
    n = 0
    t0 = time.time()
    steps_per_epoch = len(loader)

    for bi, batch in enumerate(loader):
        obs, pol_tgt = batch
        obs = obs.to(device, non_blocking=True)
        pol_tgt = pol_tgt.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(obs)
            logits = out[0] if isinstance(out, tuple) else out
            main_loss = distillation_loss(
                logits, pol_tgt,
                blend_alpha=blend_alpha,
                target_temperature=target_temperature,
                decisiveness_power=decisiveness_power)

            if aux is not None:
                lam = _aux_lambda_schedule(
                    bi, steps_per_epoch, epoch,
                    aux['target_lambda'], aux['warmup_epochs'])
                if lam > 0.0:
                    aux_obs, aux_idx = _next_aux_batch(aux)
                    # Freeze BN + use the eager module so the OOD crisis states
                    # don't poison BN running stats (which inference uses) and
                    # the freeze actually takes effect under torch.compile.
                    with frozen_bn(base_model):
                        aux_out = base_model(aux_obs)
                    aux_logits = (aux_out[0] if isinstance(aux_out, tuple)
                                  else aux_out)
                    # Anti-dilution: weight each crisis fork by its (normalized)
                    # confirmed catastrophe gap so the high-value forks dominate
                    # the aux gradient instead of being averaged into noise.
                    anchor_w = (aux['weight'][aux_idx]
                                if aux.get('weighted') else None)
                    if aux.get('mode') == 'soft':
                        # MCTS-corrections: weighted, sharpened soft-CE toward the
                        # visit distribution (the floor pipeline v2).
                        aux_loss = soft_correction_loss(
                            aux_logits,
                            aux['tgt_idx'][aux_idx],
                            aux['tgt_prob'][aux_idx],
                            anchor_weight=anchor_w,
                            target_temperature=aux['target_temperature'])
                    else:
                        aux_loss = listwise_margin_loss(
                            aux_logits,
                            aux['winner_idx'][aux_idx],
                            aux['top1_idx'][aux_idx],
                            aux['loser_idx'][aux_idx],
                            aux['loser_mask'][aux_idx],
                            margin=aux['margin'],
                            top1_weight=aux['top1_weight'],
                            other_weight=aux['other_weight'],
                            anchor_weight=anchor_w)
                    loss = main_loss + lam * aux_loss
                    total_aux += float(aux_loss.detach())
                    aux_steps += 1
                else:
                    loss = main_loss
            else:
                loss = main_loss

        # --- gradient audit: measure main vs aux gradient magnitude + alignment ---
        # Is the aux stream a gentle nudge or a sledgehammer? Run with --grad-audit N
        # (no --amp, fp32). Computes |g_main|, |g_aux|, the effective share
        # λ|g_aux|/|g_main|, and cos(g_main, g_aux) over N batches, then exits.
        if grad_audit and aux is not None and lam > 0.0:
            params = [p for p in model.parameters() if p.requires_grad]

            def _flat():
                return torch.cat([(p.grad.detach().flatten() if p.grad is not None
                                   else torch.zeros(p.numel(), device=device))
                                  for p in params])
            model.zero_grad(set_to_none=True)
            main_loss.backward()       # frees the main graph before the aux pass (mps memory)
            g_main = _flat(); n_main = g_main.norm().item()
            model.zero_grad(set_to_none=True)
            aux_loss.backward()
            g_aux = _flat(); n_aux = g_aux.norm().item()
            cos = float(torch.dot(g_main, g_aux) / (n_main * n_aux + 1e-12))
            _GRAD_AUDIT.append((n_main, n_aux, lam, cos))
            print(f"  [grad-audit {len(_GRAD_AUDIT)}/{grad_audit}] |g_main|={n_main:.4f} "
                  f"|g_aux|={n_aux:.4f} λ={lam:.3f}  λ|g_aux|/|g_main|="
                  f"{lam*n_aux/max(n_main,1e-9):.3f}  cos(main,aux)={cos:+.3f}", flush=True)
            model.zero_grad(set_to_none=True)
            if len(_GRAD_AUDIT) >= grad_audit:
                import numpy as _np, sys as _sys
                a = _np.array(_GRAD_AUDIT)
                print(f"\n=== GRAD AUDIT ({len(_GRAD_AUDIT)} batches, λ={a[:,2].mean():.3f}) ===\n"
                      f"  mean |g_main|={a[:,0].mean():.4f}  mean |g_aux|={a[:,1].mean():.4f}\n"
                      f"  effective aux share λ|g_aux|/|g_main| = {(a[:,2]*a[:,1]/a[:,0]).mean():.3f}"
                      f"   (>~0.15-0.2 => NOT a gentle nudge)\n"
                      f"  cos(main,aux) = {a[:,3].mean():+.3f}   "
                      f"(<0 aux fights main; ~0 orthogonal; >0 aligned)", flush=True)
                _sys.exit(0)
            continue   # skip the real optimizer step during the audit

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n += 1

        if (bi + 1) % log_interval == 0:
            elapsed = time.time() - t0
            sps = (bi + 1) * loader.batch_size / elapsed
            eta = (len(loader) - bi - 1) * loader.batch_size / max(sps, 1)
            aux_str = ""
            if aux is not None and aux_steps > 0:
                aux_str = (f" aux={total_aux/aux_steps:.4f} "
                           f"λ={lam:.3f}")
            print(f"  [{bi+1}/{len(loader)}] "
                  f"loss={total_loss/n:.4f}{aux_str} "
                  f"{sps:.0f} s/s ETA {eta/60:.0f}m", flush=True)

        if (aux is not None and aux['preflight_every'] > 0
                and (bi + 1) % aux['preflight_every'] == 0):
            pf = _run_soft_preflight if aux.get('mode') == 'soft' else _run_preflight
            pf(model, aux, device, amp_dtype, epoch, bi + 1, steps_per_epoch)

    return total_loss / n, (total_aux / aux_steps if aux_steps else 0.0)


def _next_aux_batch(aux):
    """Return (obs_batch, aux_indices) and advance the rotating pointer."""
    ptr_box = aux['ptr']  # mutable list holding [int]
    N = aux['obs'].size(0)
    bs = min(aux['batch_size'], N)
    start = ptr_box[0]
    end = start + bs
    if end <= N:
        idx = aux['perm'][start:end]
        ptr_box[0] = end if end < N else 0
    else:
        # Wrap: stitch tail of current perm + head of fresh perm
        if start < N:
            head = aux['perm'][start:N]
        else:
            head = aux['perm'][0:0]
        aux['perm'] = torch.randperm(N, device=aux['obs'].device)
        remaining = bs - head.numel()
        tail = aux['perm'][:remaining]
        idx = torch.cat([head, tail], dim=0)
        ptr_box[0] = remaining
    return aux['obs'][idx], idx


@torch.no_grad()
def _preflight_on(model, sub, batch_size, margin, device, amp_dtype):
    """Run preflight metrics on a sub-corpus dict (obs/winner/top1/loser/mask)."""
    use_amp = amp_dtype != torch.float32
    N = sub['obs'].size(0)
    bs = max(1, batch_size)
    chunks = []
    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
        for i in range(0, N, bs):
            out = model(sub['obs'][i:i+bs])
            chunks.append(out[0] if isinstance(out, tuple) else out)
    logits = torch.cat(chunks, dim=0).float()
    return aux_preflight_metrics(
        logits, sub['winner_idx'], sub['top1_idx'],
        sub['loser_idx'], sub['loser_mask'], margin=margin)


@torch.no_grad()
def _run_preflight(model, aux, device, amp_dtype, epoch, step_in_epoch,
                    steps_per_epoch):
    """Preflight metrics on the (train) aux corpus and the held-out split.

    The held-out fork metrics are the load-bearing signal: pillar3c regressed
    while train metrics looked fine, so we watch whether the safe-move ranking
    generalizes to forks from UNSEEN games (project_pillar3c_failure.md).
    """
    was_training = model.training
    model.train(False)

    def _fmt(tag, m):
        return (f"  [{tag} ep{epoch+1} step {step_in_epoch}/{steps_per_epoch}] "
                f"flip={m['stored_top1_flip_rate']:.3f} "
                f"margin(win-pol)={m['mean_top1_margin']:+.2f} "
                f"conc={m['clean_loser_concordance']:.3f} "
                f"clean@{aux['margin']}={m['all_clean_loser_margin_rate']:.3f}")

    m = _preflight_on(model, aux, aux['batch_size'], aux['margin'],
                      device, amp_dtype)
    print(_fmt('preflight', m), flush=True)
    if aux.get('heldout') is not None:
        mh = _preflight_on(model, aux['heldout'], aux['batch_size'],
                           aux['margin'], device, amp_dtype)
        print(_fmt('heldout ', mh), flush=True)

    if (aux.get('abort_flip_rate') is not None
            and step_in_epoch >= aux.get('abort_after_step', 0)
            and m['stored_top1_flip_rate'] < aux['abort_flip_rate']):
        if was_training:
            model.train(True)
        raise SystemExit(
            f"[ABORT] stored_top1_flip_rate {m['stored_top1_flip_rate']:.4f} "
            f"< threshold {aux['abort_flip_rate']} at step "
            f"{step_in_epoch} epoch {epoch+1}. Increase --aux-lambda "
            f"or relax filter, then re-run.")
    if was_training:
        model.train(True)


@torch.no_grad()
def _run_soft_preflight(model, aux, device, amp_dtype, epoch, step_in_epoch,
                        steps_per_epoch):
    """Preflight for MCTS-corrections (soft mode): match-rate (policy argmax ==
    MCTS top move) + weighted soft-CE, on train and held-out. Held-out match-rate
    rising = the corrections GENERALIZE; flat = memorizing (Track-1's failure)."""
    was_training = model.training
    model.train(False)
    use_amp = amp_dtype != torch.float32

    def _on(sub):
        N, bs = sub['obs'].size(0), max(1, aux['batch_size'])
        match, ce_sum = 0, 0.0
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            for i in range(0, N, bs):
                out = model(sub['obs'][i:i + bs])
                logits = (out[0] if isinstance(out, tuple) else out).float()
                ti, tp = sub['tgt_idx'][i:i + bs], sub['tgt_prob'][i:i + bs]
                match += (logits.argmax(1) == ti[:, 0]).sum().item()
                logp = torch.log_softmax(logits, 1)
                ce_sum += (-(tp * torch.gather(logp, 1, ti)).sum(1)).sum().item()
        return match / max(N, 1), ce_sum / max(N, 1)

    mr, ce = _on(aux)
    print(f"  [soft-preflight ep{epoch+1} step {step_in_epoch}/{steps_per_epoch}] "
          f"match(argmax=MCTS)={mr:.3f} softCE={ce:.3f}", flush=True)
    if aux.get('heldout') is not None:
        mrh, ceh = _on(aux['heldout'])
        print(f"  [soft-heldout  ep{epoch+1} step {step_in_epoch}/{steps_per_epoch}] "
              f"match={mrh:.3f} softCE={ceh:.3f}", flush=True)
    if was_training:
        model.train(True)


@torch.no_grad()
def validate(model, loader, device, amp_dtype=torch.float32):
    """Validation on un-sharpened targets (faithful CE to V12)."""
    model.train(False)
    use_amp = amp_dtype != torch.float32
    total = 0.0
    n = 0
    for obs, pol_tgt in loader:
        obs = obs.to(device, non_blocking=True)
        pol_tgt = pol_tgt.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(obs)
            logits = out[0] if isinstance(out, tuple) else out
            loss = cross_entropy_soft(logits, pol_tgt)
        total += loss.item()
        n += 1
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor-file', required=True)
    p.add_argument('--epochs', type=int, default=17)
    p.add_argument('--batch-size', type=int, default=32768)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup-epochs', type=int, default=1)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--val-split', type=float, default=0.05)
    # Augmentation defaults: BOTH ON. Color permutation (7! symmetry) was
    # the +4% lift in Path B v1 (HISTORY 143). Dihedral 8× is standard.
    p.add_argument('--no-color-augment', action='store_true',
                   help='Disable color permutation augmentation (default ON).')
    p.add_argument('--no-dihedral-augment', action='store_true',
                   help='Disable 8x dihedral augmentation (default ON).')
    p.add_argument('--augment-factor', type=int, default=8,
                   help='Epoch length multiplier when augmenting. Each '
                        '"extra" pass over the same base state gets a '
                        'different random transform.')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--grad-audit', type=int, default=0,
                   help='Diagnostic: measure main vs aux gradient norm + cosine over N batches '
                        'then exit (run without --amp for fp32). 0=off.')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile for faster forward/backward')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--warm-start', action='store_true',
                   help='Load weights only; reset optimizer/scheduler.')
    p.add_argument('--save-dir', default='checkpoints/run')
    p.add_argument('--copy-to', type=str, default=None,
                   help='Path to copy best.pt for off-machine backup.')
    # Loss-shape knobs
    p.add_argument('--target-temperature', type=float, default=1.0,
                   help='Sharpen targets via t**(1/T) renormalized. '
                        '1.0=no change. 0.5=produced sharp_50 win (HISTORY 154).')
    p.add_argument('--decisiveness-power', type=float, default=0.0,
                   help='Weight each state CE by (visit top-share)**power, mean-1 '
                        'normalized. >0 makes DECISIVE escape states drive the gradient '
                        'and flat quiet states near-zero weight (so they cannot de-peak '
                        'a strong base). Try 3.0. 0=off (uniform).')
    p.add_argument('--blend-alpha', type=float, default=1.0,
                   help='Convex blend of soft CE and hard argmax CE. 1.0=soft '
                        'only (default). Rarely useful.')
    # Phase 3 counterfactual auxiliary loss (see alphatrain/counterfactual.py)
    p.add_argument('--aux-counterfactual', type=str, default=None,
                   help='Path to stationary_counterfactuals_v1.pt. When set, '
                        'adds the listwise-margin auxiliary loss for floor-'
                        'aware distillation (Phase 3).')
    p.add_argument('--aux-lambda', type=float, default=0.20,
                   help='Target λ for the aux loss after warmup.')
    p.add_argument('--aux-margin', type=float, default=AUX_DEFAULT_MARGIN,
                   help='Hinge margin in listwise loss (default 0.25).')
    p.add_argument('--aux-top1-weight', type=float,
                   default=AUX_DEFAULT_TOP1_WEIGHT,
                   help='Weight on stored-top1 pair (default 1.0).')
    p.add_argument('--aux-other-weight', type=float,
                   default=AUX_DEFAULT_OTHER_WEIGHT,
                   help='Weight on clean-other-loser pairs (default 0.5).')
    p.add_argument('--aux-batch-size', type=int, default=256,
                   help='Aux anchors per main step. With ~528 anchors and '
                        '256 batch, the corpus cycles every ~2 steps.')
    p.add_argument('--aux-warmup-epochs', type=float, default=2.0,
                   help='λ linearly ramps 0→target over this many epochs.')
    p.add_argument('--aux-preflight-every', type=int, default=100,
                   help='Main-step interval for preflight metrics print.')
    p.add_argument('--aux-abort-flip-rate', type=float, default=None,
                   help='If set, abort training when preflight reports '
                        'stored_top1_flip_rate below this threshold '
                        '(checked after --aux-abort-after-step).')
    p.add_argument('--aux-abort-after-step', type=int, default=500,
                   help='Earliest step (within an epoch) at which abort can '
                        'trigger. Default 500 matches Phase 3 plan.')
    # Crisis-fork aux loss (the floor work). Same listwise-margin machinery as
    # --aux-counterfactual, but fed CI-confirmed R=500 forks from the
    # rewind-from-death harvest (logs/mine_*.json) instead of R=24 stationary
    # counterfactuals. These two flags are mutually exclusive.
    p.add_argument('--aux-crisis', type=str, default=None,
                   help='Glob of mine_crisis_sweep outputs (e.g. '
                        '"logs/mine_*.json"). Builds the confirmed-fork aux '
                        'corpus and adds the floor-aware listwise-margin loss.')
    p.add_argument('--aux-crisis-corpus', type=str, default=None,
                   help='Path to a PRE-BUILT crisis corpus .pt '
                        '(scripts/build_crisis_corpus_file.py). Use this on '
                        'Colab instead of --aux-crisis so you upload one small '
                        'file, not the mine/death JSON tree.')
    p.add_argument('--aux-corrections-corpus', type=str, default=None,
                   help='Path to a PRE-BUILT MCTS-corrections corpus .pt '
                        '(scripts/build_corrections_corpus.py). Floor pipeline v2: '
                        'weighted, sharpened soft-CE toward the MCTS visit '
                        'distribution. Mutually exclusive with the other aux modes.')
    p.add_argument('--aux-target-temperature', type=float, default=0.5,
                   help='Sharpening for the MCTS-corrections soft target '
                        '(<1 = commit to the high-visit move). Independent of the '
                        'main --target-temperature.')
    p.add_argument('--aux-crisis-death-dir',
                   default='alphatrain/data/death_games',
                   help='Directory of death_{seed}.json (for the policy move).')
    p.add_argument('--aux-clean-loser-margin', type=float, default=10.0,
                   help='A candidate is a clean loser only if its catastrophe '
                        'exceeds the winner by >= this many pp (noise guard; '
                        'R=100 gap SE ~7pp).')
    p.add_argument('--aux-min-gap', type=float, default=0.0,
                   help='Drop confirmed forks whose catastrophe gap is below '
                        'this (pp). 0 = keep all CI-confirmed forks.')
    p.add_argument('--aux-weighted', action='store_true',
                   help='Weight each fork by its normalized confirmed gap in '
                        'the listwise loss (anti-dilution; high-gap forks '
                        'dominate). Off = uniform.')
    p.add_argument('--aux-holdout-frac', type=float, default=0.0,
                   help='Fraction of SEEDS held out of aux training and '
                        'monitored separately (generalization check). 0=off.')
    p.add_argument('--aux-split-seed', type=int, default=0,
                   help='RNG seed for the by-seed held-out split.')
    p.add_argument('--max-train-states', type=int, default=0,
                   help='Cap on base training states (local smoke only; '
                        '0 = use all). Subsamples the main corpus, not the aux.')
    args = p.parse_args()
    if sum(bool(x) for x in (args.aux_counterfactual,
                             args.aux_crisis or args.aux_crisis_corpus,
                             args.aux_corrections_corpus)) > 1:
        raise SystemExit("Pick ONE aux mode: --aux-counterfactual, "
                         "--aux-crisis[-corpus], or --aux-corrections-corpus.")

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    # Data
    color_augment = not args.no_color_augment
    dihedral_augment = not args.no_dihedral_augment
    print(f"Augmentation: color={color_augment} dihedral={dihedral_augment} "
          f"factor={args.augment_factor}", flush=True)

    train_set, val_set = TensorDatasetGPU.make_train_val_split(
        args.tensor_file,
        val_split=args.val_split,
        augment=dihedral_augment,
        color_augment=color_augment,
        augment_factor=args.augment_factor,
        device=str(device),
        seed=42,
    )
    if args.max_train_states and args.max_train_states < len(train_set.base_indices):
        train_set.base_indices = train_set.base_indices[:args.max_train_states]
        print(f"  SUBSAMPLE: train base states capped to "
              f"{len(train_set.base_indices):,} (local smoke)", flush=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0,
                              collate_fn=train_set.collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=0,
                            collate_fn=val_set.collate)
    print(f"Train={len(train_set):,}  Val={len(val_set):,}", flush=True)

    # Model
    model = AlphaTrainNet(num_blocks=args.num_blocks,
                          channels=args.channels).to(device)
    n_params = count_parameters(model)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    print(f"Model: {args.num_blocks}b x {args.channels}ch, "
          f"{n_params:,} params", flush=True)

    # Resume / warm start (load BEFORE compile to avoid prefix issues)
    start_epoch = 0
    best_val = float('inf')
    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(f"--resume not found: {args.resume}")
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device,
                          weights_only=False)
        state = ckpt['model']
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in state if k not in filtered]
        model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"  Skipped {len(skipped)} keys (shape/name mismatch)",
                  flush=True)
        if not args.warm_start and 'optimizer' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            best_val = ckpt.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch - 1}", flush=True)
        else:
            print(f"Warm-start from epoch {ckpt.get('epoch', '?')}, "
                  f"fresh optimizer", flush=True)

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("torch.compile enabled", flush=True)

    # Optimizer / scheduler
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(train_params, lr=args.lr,
                                   weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    if (args.resume and not args.warm_start and ckpt is not None
            and 'optimizer' in ckpt):
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    # AMP
    use_amp = args.amp and device.type == 'cuda'
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = None
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler('cuda')
    else:
        amp_dtype = torch.float32
        scaler = None
    if use_amp:
        print(f"AMP enabled (dtype={amp_dtype})", flush=True)

    if args.target_temperature != 1.0:
        print(f"Target sharpening: T={args.target_temperature} "
              f"(targets**(1/{args.target_temperature}))", flush=True)
    if args.decisiveness_power > 0.0:
        print(f"Decisiveness weighting: (top-share)**{args.decisiveness_power} "
              f"(decisive escape states drive the gradient; flat quiet states "
              f"near-zero weight)", flush=True)
    if args.blend_alpha != 1.0:
        print(f"Blended CE: alpha={args.blend_alpha} "
              f"(blend_alpha*soft + (1-blend_alpha)*hard)", flush=True)

    # Phase 3 counterfactual aux corpus setup
    aux = None
    if args.aux_counterfactual:
        print(f"\nLoading aux corpus: {args.aux_counterfactual}", flush=True)
        corpus = build_aux_corpus(args.aux_counterfactual, device=str(device))
        N_aux = corpus['boards'].size(0)
        n_pairs = (N_aux + int(corpus['loser_mask'].sum().item()))
        print(f"  Aux: {N_aux} anchors, {n_pairs} pairs "
              f"(filter: die≥{1/24:.4f} OR leave≥{2/24:.4f})", flush=True)

        # Build canonical observations once. Use train_set's GPU obs builder
        # so the layout matches the main batch exactly.
        with torch.no_grad():
            aux_obs = train_set._build_obs_core(
                corpus['boards'].long(),
                next_pos=corpus['next_pos'],
                next_col=corpus['next_col'],
                n_next=corpus['n_next'])
            if device.type == 'cuda':
                aux_obs = aux_obs.contiguous(
                    memory_format=torch.channels_last)
        aux = {
            'obs': aux_obs,
            'winner_idx': corpus['winner_idx'],
            'top1_idx': corpus['top1_idx'],
            'loser_idx': corpus['loser_idx'],
            'loser_mask': corpus['loser_mask'],
            'batch_size': args.aux_batch_size,
            'target_lambda': args.aux_lambda,
            'margin': args.aux_margin,
            'top1_weight': args.aux_top1_weight,
            'other_weight': args.aux_other_weight,
            'warmup_epochs': args.aux_warmup_epochs,
            'preflight_every': args.aux_preflight_every,
            'abort_flip_rate': args.aux_abort_flip_rate,
            'abort_after_step': args.aux_abort_after_step,
            'perm': torch.randperm(N_aux, device=device),
            'ptr': [0],
        }
        print(f"  λ target={args.aux_lambda} margin={args.aux_margin} "
              f"top1_w={args.aux_top1_weight} other_w={args.aux_other_weight} "
              f"warmup_ep={args.aux_warmup_epochs} batch={args.aux_batch_size}",
              flush=True)
        # Baseline preflight before any optimization steps.
        _run_preflight(model, aux, device, amp_dtype,
                       epoch=-1, step_in_epoch=0,
                       steps_per_epoch=len(train_loader))

    # Crisis-fork aux corpus setup (the floor work)
    if args.aux_crisis or args.aux_crisis_corpus:
        if args.aux_crisis_corpus:
            print(f"\nLoading pre-built crisis corpus: "
                  f"{args.aux_crisis_corpus}", flush=True)
            raw = torch.load(args.aux_crisis_corpus, map_location='cpu',
                             weights_only=False)
            corpus = {k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in raw.items()}
        else:
            print(f"\nBuilding crisis-fork aux corpus: {args.aux_crisis}",
                  flush=True)
            corpus = build_crisis_corpus(
                args.aux_crisis, death_dir=args.aux_crisis_death_dir,
                device=str(device),
                clean_loser_margin=args.aux_clean_loser_margin,
                min_gap=args.aux_min_gap)
        cs = corpus['_stats']
        print(f"  {cs['n_anchors']} confirmed forks from {cs['n_seeds']} seeds "
              f"/ {cs['n_files']} games; {cs['n_clean_pairs']} clean-loser "
              f"pairs (dropped {cs['n_unconfirmed']} unconfirmed, "
              f"{cs['n_degenerate']} degenerate)", flush=True)

        with torch.no_grad():
            full_obs = train_set._build_obs_core(
                corpus['boards'].long(), next_pos=corpus['next_pos'],
                next_col=corpus['next_col'], n_next=corpus['n_next'])
            if device.type == 'cuda':
                full_obs = full_obs.contiguous(
                    memory_format=torch.channels_last)

        # By-seed held-out split (anchors from one game are correlated; split
        # by seed so held-out forks genuinely test generalization).
        seeds_t = corpus['seed']
        uniq = sorted(set(int(s) for s in seeds_t.tolist()))
        n_hold = (max(1, int(round(args.aux_holdout_frac * len(uniq))))
                  if args.aux_holdout_frac > 0 else 0)
        if n_hold > 0:
            g = torch.Generator().manual_seed(args.aux_split_seed)
            hperm = torch.randperm(len(uniq), generator=g).tolist()
            hold_seeds = set(uniq[i] for i in hperm[:n_hold])
            hold_mask = torch.tensor([int(s) in hold_seeds
                                      for s in seeds_t.tolist()],
                                     device=device)
        else:
            hold_mask = torch.zeros(len(seeds_t), dtype=torch.bool,
                                    device=device)
        tr = (~hold_mask).nonzero(as_tuple=True)[0]
        ho = hold_mask.nonzero(as_tuple=True)[0]
        # Renormalize train weights to mean 1 over the TRAIN subset so λ keeps
        # its meaning after the held-out anchors are removed.
        w_tr = corpus['weight'][tr]
        w_tr = w_tr / w_tr.mean().clamp(min=1e-6)
        N_aux = tr.numel()
        aux = {
            'obs': full_obs[tr],
            'winner_idx': corpus['winner_idx'][tr],
            'top1_idx': corpus['top1_idx'][tr],
            'loser_idx': corpus['loser_idx'][tr],
            'loser_mask': corpus['loser_mask'][tr],
            'weight': w_tr,
            'weighted': args.aux_weighted,
            'batch_size': args.aux_batch_size,
            'target_lambda': args.aux_lambda,
            'margin': args.aux_margin,
            'top1_weight': args.aux_top1_weight,
            'other_weight': args.aux_other_weight,
            'warmup_epochs': args.aux_warmup_epochs,
            'preflight_every': args.aux_preflight_every,
            'abort_flip_rate': args.aux_abort_flip_rate,
            'abort_after_step': args.aux_abort_after_step,
            'perm': torch.randperm(N_aux, device=device),
            'ptr': [0],
        }
        if ho.numel() > 0:
            aux['heldout'] = {
                'obs': full_obs[ho],
                'winner_idx': corpus['winner_idx'][ho],
                'top1_idx': corpus['top1_idx'][ho],
                'loser_idx': corpus['loser_idx'][ho],
                'loser_mask': corpus['loser_mask'][ho],
            }
        print(f"  train forks={N_aux} heldout forks={ho.numel()} "
              f"({n_hold}/{len(uniq)} seeds); weighted={args.aux_weighted} "
              f"λ={args.aux_lambda} margin={args.aux_margin} "
              f"warmup_ep={args.aux_warmup_epochs} batch={args.aux_batch_size}",
              flush=True)
        _run_preflight(model, aux, device, amp_dtype,
                       epoch=-1, step_in_epoch=0,
                       steps_per_epoch=len(train_loader))

    # MCTS-corrections aux setup (floor pipeline v2: weighted/sharpened soft-CE)
    if args.aux_corrections_corpus:
        print(f"\nLoading MCTS-corrections corpus: "
              f"{args.aux_corrections_corpus}", flush=True)
        raw = torch.load(args.aux_corrections_corpus, map_location='cpu',
                         weights_only=False)
        corpus = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in raw.items()}
        cs = corpus['_stats']
        print(f"  {cs['n_corrections']} corrections from {cs['n_seeds']} seeds "
              f"/ {cs['n_files']} games", flush=True)
        with torch.no_grad():
            full_obs = train_set._build_obs_core(
                corpus['boards'].long(), next_pos=corpus['next_pos'],
                next_col=corpus['next_col'], n_next=corpus['n_next'])
            if device.type == 'cuda':
                full_obs = full_obs.contiguous(memory_format=torch.channels_last)
        seeds_t = corpus['seed']
        uniq = sorted(set(int(s) for s in seeds_t.tolist()))
        n_hold = (max(1, int(round(args.aux_holdout_frac * len(uniq))))
                  if args.aux_holdout_frac > 0 else 0)
        if n_hold > 0:
            g = torch.Generator().manual_seed(args.aux_split_seed)
            hperm = torch.randperm(len(uniq), generator=g).tolist()
            hold_seeds = set(uniq[i] for i in hperm[:n_hold])
            hold_mask = torch.tensor([int(s) in hold_seeds
                                      for s in seeds_t.tolist()], device=device)
        else:
            hold_mask = torch.zeros(len(seeds_t), dtype=torch.bool, device=device)
        tr = (~hold_mask).nonzero(as_tuple=True)[0]
        ho = hold_mask.nonzero(as_tuple=True)[0]
        w_tr = corpus['weight'][tr]
        w_tr = w_tr / w_tr.mean().clamp(min=1e-6)
        N_aux = tr.numel()
        aux = {
            'mode': 'soft',
            'obs': full_obs[tr],
            'tgt_idx': corpus['tgt_idx'][tr],
            'tgt_prob': corpus['tgt_prob'][tr],
            'weight': w_tr,
            'weighted': args.aux_weighted,
            'target_temperature': args.aux_target_temperature,
            'batch_size': args.aux_batch_size,
            'target_lambda': args.aux_lambda,
            'warmup_epochs': args.aux_warmup_epochs,
            'preflight_every': args.aux_preflight_every,
            'perm': torch.randperm(N_aux, device=device),
            'ptr': [0],
        }
        if ho.numel() > 0:
            aux['heldout'] = {'obs': full_obs[ho],
                              'tgt_idx': corpus['tgt_idx'][ho],
                              'tgt_prob': corpus['tgt_prob'][ho]}
        print(f"  train corrections={N_aux} heldout={ho.numel()} "
              f"({n_hold}/{len(uniq)} seeds); weighted={args.aux_weighted} "
              f"λ={args.aux_lambda} aux_T={args.aux_target_temperature} "
              f"warmup_ep={args.aux_warmup_epochs} batch={args.aux_batch_size}",
              flush=True)
        _run_soft_preflight(model, aux, device, amp_dtype,
                            epoch=-1, step_in_epoch=0,
                            steps_per_epoch=len(train_loader))

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\n=== Training {args.epochs} epochs ===", flush=True)
    t_total = time.time()

    for epoch in range(start_epoch, args.epochs):
        et = time.time()
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.2e})", flush=True)

        tl, aux_tl = train_epoch(model, train_loader, optimizer, device,
                                  scaler, amp_dtype,
                                  blend_alpha=args.blend_alpha,
                                  target_temperature=args.target_temperature,
                                  aux=aux, epoch=epoch, grad_audit=args.grad_audit,
                                  decisiveness_power=args.decisiveness_power)
        vl = validate(model, val_loader, device, amp_dtype=amp_dtype)
        scheduler.step()

        print(f"  Train: loss={tl:.4f}"
              + (f"  aux_loss={aux_tl:.4f}" if aux is not None else ""),
              flush=True)
        print(f"  V12 val: loss={vl:.4f} [{time.time()-et:.0f}s]",
              flush=True)
        if aux is not None:
            pf = _run_soft_preflight if aux.get('mode') == 'soft' else _run_preflight
            pf(model, aux, device, amp_dtype, epoch=epoch,
               step_in_epoch=len(train_loader), steps_per_epoch=len(train_loader))

        ck = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': vl,
            'best_val_loss': min(best_val, vl),
            'args': vars(args),
            'policy_only': True,
        }
        latest = os.path.join(args.save_dir, 'latest.pt')
        torch.save(ck, latest)
        epoch_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pt')
        torch.save(ck, epoch_path)
        if args.copy_to:
            copy_dir = os.path.dirname(args.copy_to) or '.'
            run_prefix = (os.path.basename(args.save_dir.rstrip('/'))
                          or 'run')
            shutil.copy2(latest,
                          os.path.join(copy_dir, f'{run_prefix}_latest.pt'))
            shutil.copy2(epoch_path,
                          os.path.join(copy_dir,
                                        f'{run_prefix}_epoch_{epoch+1}.pt'))
        if vl < best_val:
            best_val = vl
            best_path = os.path.join(args.save_dir, 'best.pt')
            torch.save(ck, best_path)
            print(f"  ** New best (val={vl:.4f}) **", flush=True)
            if args.copy_to:
                shutil.copy2(best_path, args.copy_to)

    print(f"\n=== Done: {args.epochs} epochs in "
          f"{(time.time()-t_total)/3600:.1f}h ===", flush=True)
    print(f"Best V12 val: {best_val:.4f}", flush=True)


if __name__ == '__main__':
    main()
