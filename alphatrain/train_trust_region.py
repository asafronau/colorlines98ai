"""Trust-region distillation of MCTS crisis corrections (pillar3e).

Structural successor to the aux-loss recipe in train_path_b (pillar3d). There the
4800-sim corrections were an auxiliary exception list bolted onto a re-distill of
the stale 400-sim V13 targets — λ in [0.01..0.014] was empirically flat, ep3+
always drifted (the converged base's noise-floor gradient accumulates), and the
channel plateaued. Here the corrections ARE the teacher, and general play is
preserved by a trust region instead of re-teaching old targets:

    loss = teacher_CE(student, corrections)  +  β · KL(frozen ‖ student) on broad states

  * teacher_CE: weighted, sharpened soft-CE toward the 4800-sim visit
    distribution (same target machinery as the mC recipe), frozen-BN forward
    (crisis states are OOD; see frozen_bn).
  * anchor: KL(frozen pillar3b ‖ student) on V13 self-play states. EXACTLY ZERO
    at the warm start — no force until the policy actually moves, never pulls
    toward stale targets, so long training is stable and exposure scales with
    the corpus. Anchor states need no labels (only obs), so they can later be
    refreshed on-policy for free.

Epoch = one pass over the corrections (the corpus is small: ~20k → ~16 steps at
batch 1024); anchor batches are sampled per step from the V13 tensor. Run many
epochs; checkpoints every --save-every.

β is set empirically: --grad-audit-every prints |g_teacher|, |g_anchor| and the
effective anchor share β|g_anchor|/|g_teacher| DURING training (an audit at init
is useless — the anchor gradient is zero there). Sweep β to hit teacher:anchor
gradient ratios ≈ 3:1 / 1:1 / 1:3.

Usage (arm A — visit-softmax targets, existing corpus):
    python -m alphatrain.train_trust_region \\
        --corrections-corpus crisis/corrections_corpus_mm05.pt \\
        --anchor-tensor alphatrain/data/v13_pillar3a.pt \\
        --frozen alphatrain/data/pillar3b_epoch_20.pt \\
        --beta 1.0 --epochs 60 --lr 5e-5 \\
        --teacher-batch-size 1024 --anchor-batch-size 4096 \\
        --target-temperature 0.5 --weighted \\
        --holdout-frac 0.15 --split-seed 0 \\
        --save-dir checkpoints/pillar3e_tr --save-every 5
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

from alphatrain.counterfactual import soft_correction_loss
from alphatrain.dataset import TensorDatasetGPU
from alphatrain.model import AlphaTrainNet, count_parameters
from alphatrain.train_path_b import cross_entropy_soft, frozen_bn


def anchor_kl(frozen_logits, student_logits):
    """KL(frozen ‖ student), mean over the batch. Zero when student == frozen."""
    logp_f = F.log_softmax(frozen_logits.float(), dim=-1)
    logp_s = F.log_softmax(student_logits.float(), dim=-1)
    return (logp_f.exp() * (logp_f - logp_s)).sum(dim=-1).mean()


def load_policy_state(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    return state, ckpt


def build_corrections(corpus_path, train_set, device, holdout_frac, split_seed):
    """Load a corrections corpus .pt, build canonical obs, split by SEED.

    Same semantics as train_path_b's --aux-corrections-corpus: anchors from one
    game are correlated, so the held-out split is by seed; train weights are
    renormalized to mean 1 over the train subset.
    """
    raw = torch.load(corpus_path, map_location='cpu', weights_only=False)
    corpus = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in raw.items()}
    cs = corpus['_stats']
    print(f"  {cs['n_corrections']} corrections from {cs['n_seeds']} seeds "
          f"/ {cs['n_files']} games (min_margin={cs['min_margin']})", flush=True)
    with torch.no_grad():
        full_obs = train_set._build_obs_core(
            corpus['boards'].long(), next_pos=corpus['next_pos'],
            next_col=corpus['next_col'], n_next=corpus['n_next'])
        if device.type == 'cuda':
            full_obs = full_obs.contiguous(memory_format=torch.channels_last)
    seeds_t = corpus['seed']
    uniq = sorted(set(int(s) for s in seeds_t.tolist()))
    n_hold = (max(1, int(round(holdout_frac * len(uniq))))
              if holdout_frac > 0 else 0)
    if n_hold > 0:
        g = torch.Generator().manual_seed(split_seed)
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
    train = {'obs': full_obs[tr], 'tgt_idx': corpus['tgt_idx'][tr],
             'tgt_prob': corpus['tgt_prob'][tr], 'weight': w_tr}
    heldout = None
    if ho.numel() > 0:
        heldout = {'obs': full_obs[ho], 'tgt_idx': corpus['tgt_idx'][ho],
                   'tgt_prob': corpus['tgt_prob'][ho]}
    print(f"  train={tr.numel()} heldout={ho.numel()} "
          f"({n_hold}/{len(uniq)} seeds held out)", flush=True)
    return train, heldout


@torch.no_grad()
def corrections_metrics(model, sub, batch_size, device, amp_dtype):
    """match-rate (argmax == MCTS top) + unweighted softCE on a corpus split."""
    was_training = model.training
    model.train(False)
    use_amp = amp_dtype != torch.float32
    N, bs = sub['obs'].size(0), max(1, batch_size)
    match, ce_sum = 0, 0.0
    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
        for i in range(0, N, bs):
            out = model(sub['obs'][i:i + bs])
            logits = (out[0] if isinstance(out, tuple) else out).float()
            ti, tp = sub['tgt_idx'][i:i + bs], sub['tgt_prob'][i:i + bs]
            match += (logits.argmax(1) == ti[:, 0]).sum().item()
            logp = torch.log_softmax(logits, 1)
            ce_sum += (-(tp * torch.gather(logp, 1, ti)).sum(1)).sum().item()
    model.train(was_training)
    return match / max(N, 1), ce_sum / max(N, 1)


@torch.no_grad()
def validate_drift(model, frozen, val_loader, device, amp_dtype):
    """V13 val CE (vs stored targets) + mean anchor KL vs frozen — the two
    drift gauges. KL is the direct trust-region readout."""
    was_training = model.training
    model.train(False)
    use_amp = amp_dtype != torch.float32
    ce_tot, kl_tot, n = 0.0, 0.0, 0
    for obs, pol_tgt in val_loader:
        obs = obs.to(device, non_blocking=True)
        pol_tgt = pol_tgt.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(obs)
            s_logits = out[0] if isinstance(out, tuple) else out
            f_out = frozen(obs)
            f_logits = f_out[0] if isinstance(f_out, tuple) else f_out
        ce_tot += float(cross_entropy_soft(s_logits.float(), pol_tgt))
        kl_tot += float(anchor_kl(f_logits, s_logits))
        n += 1
    model.train(was_training)
    return ce_tot / n, kl_tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corrections-corpus', required=True,
                   help='Pre-built corpus .pt (scripts/build_corrections_corpus.py).')
    p.add_argument('--anchor-tensor', required=True,
                   help='Broad-state tensor (e.g. v13_pillar3a.pt). Only the obs '
                        'are used for the KL anchor; targets feed the val CE gauge.')
    p.add_argument('--frozen', required=True,
                   help='Checkpoint of the trust-region center (pillar3b_epoch_20). '
                        'Student warm-starts from it unless --resume is given.')
    p.add_argument('--resume', default=None,
                   help='Warm-start the STUDENT from this checkpoint instead of '
                        '--frozen (the anchor stays --frozen).')
    p.add_argument('--beta', type=float, default=1.0,
                   help='Anchor strength. Tune via --grad-audit-every to hit '
                        'teacher:anchor gradient ratios ~3:1 / 1:1 / 1:3.')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--teacher-batch-size', type=int, default=1024)
    p.add_argument('--anchor-batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--target-temperature', type=float, default=0.5,
                   help='Sharpening for the visit-distribution target (mC keeper).')
    p.add_argument('--weighted', action='store_true',
                   help='Margin-weight the corrections (mC keeper).')
    p.add_argument('--holdout-frac', type=float, default=0.15)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--val-split', type=float, default=0.02,
                   help='Anchor-tensor val fraction for the drift gauges.')
    p.add_argument('--no-color-augment', action='store_true')
    p.add_argument('--no-dihedral-augment', action='store_true')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--grad-audit-every', type=int, default=50,
                   help='Every N steps print |g_teacher|, |g_anchor| and the '
                        'effective anchor share β|g_anchor|/|g_teacher|. 0=off.')
    p.add_argument('--max-anchor-states', type=int, default=0,
                   help='Cap anchor base states (local smoke only; 0=all).')
    p.add_argument('--save-dir', default='checkpoints/pillar3e_tr')
    p.add_argument('--save-every', type=int, default=5)
    p.add_argument('--copy-to', default=None,
                   help='Drive path: epoch checkpoints are mirrored next to it.')
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    # Anchor states (broad distribution). Augmentation on by default — the
    # anchor holds the policy near frozen across the augmented play
    # distribution, and the frozen forward sees the same augmented obs.
    train_set, val_set = TensorDatasetGPU.make_train_val_split(
        args.anchor_tensor, val_split=args.val_split,
        augment=not args.no_dihedral_augment,
        color_augment=not args.no_color_augment,
        augment_factor=1, device=str(device), seed=42)
    if args.max_anchor_states and args.max_anchor_states < len(train_set.base_indices):
        train_set.base_indices = train_set.base_indices[:args.max_anchor_states]
        print(f"  SUBSAMPLE: anchor base states capped to "
              f"{len(train_set.base_indices):,} (local smoke)", flush=True)
    anchor_loader = DataLoader(train_set, batch_size=args.anchor_batch_size,
                               shuffle=True, num_workers=0,
                               collate_fn=train_set.collate)
    val_loader = DataLoader(val_set, batch_size=args.anchor_batch_size * 2,
                            shuffle=False, num_workers=0,
                            collate_fn=val_set.collate)
    print(f"Anchor states={len(train_set):,}  val={len(val_set):,}", flush=True)

    # Frozen trust-region center + student.
    frozen_state, _ = load_policy_state(args.frozen, device)
    nb = sum(1 for k in frozen_state
             if k.endswith('.conv1.weight') and k.startswith('blocks.'))
    ch = frozen_state['stem.0.weight'].shape[0]
    frozen = AlphaTrainNet(num_blocks=nb, channels=ch).to(device)
    frozen.load_state_dict(frozen_state, strict=False)
    frozen.train(False)
    frozen.requires_grad_(False)
    model = AlphaTrainNet(num_blocks=nb, channels=ch).to(device)
    student_state = frozen_state
    if args.resume:
        student_state, _ = load_policy_state(args.resume, device)
        print(f"Student warm-start: {args.resume}", flush=True)
    model.load_state_dict(student_state, strict=False)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        frozen = frozen.to(memory_format=torch.channels_last)
    print(f"Model: {nb}b x {ch}ch, {count_parameters(model):,} params  "
          f"(frozen anchor: {args.frozen})", flush=True)

    print(f"\nLoading corrections corpus: {args.corrections_corpus}", flush=True)
    corr, corr_held = build_corrections(
        args.corrections_corpus, train_set, device,
        args.holdout_frac, args.split_seed)
    N_corr = corr['obs'].size(0)
    steps_per_epoch = (N_corr + args.teacher_batch_size - 1) // args.teacher_batch_size

    base_model = model
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("torch.compile enabled (student)", flush=True)

    optimizer = torch.optim.AdamW(
        [q for q in model.parameters() if q.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    use_amp = args.amp and device.type == 'cuda'
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype, scaler = torch.bfloat16, None
    elif use_amp:
        amp_dtype, scaler = torch.float16, torch.amp.GradScaler('cuda')
    else:
        amp_dtype, scaler = torch.float32, None
    if use_amp:
        print(f"AMP enabled (dtype={amp_dtype})", flush=True)

    mr0, ce0 = corrections_metrics(base_model, corr, args.teacher_batch_size,
                                   device, amp_dtype)
    line = f"\n[baseline] corr match={mr0:.3f} softCE={ce0:.3f}"
    if corr_held is not None:
        mrh, ceh = corrections_metrics(base_model, corr_held,
                                       args.teacher_batch_size, device, amp_dtype)
        line += f" | heldout match={mrh:.3f} softCE={ceh:.3f}"
    print(line, flush=True)
    print(f"β={args.beta} T={args.target_temperature} weighted={args.weighted} "
          f"teacher_bs={args.teacher_batch_size} anchor_bs={args.anchor_batch_size} "
          f"steps/ep={steps_per_epoch}", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    anchor_iter = iter(anchor_loader)
    print(f"\n=== Training {args.epochs} epochs "
          f"({args.epochs * steps_per_epoch} steps) ===", flush=True)
    t_total = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        et = time.time()
        model.train(True)
        perm = torch.randperm(N_corr, device=device)
        tl_sum = kl_sum = 0.0
        n_steps = 0
        for bi in range(steps_per_epoch):
            idx = perm[bi * args.teacher_batch_size:
                       (bi + 1) * args.teacher_batch_size]
            try:
                anchor_obs, _ = next(anchor_iter)
            except StopIteration:
                anchor_iter = iter(anchor_loader)
                anchor_obs, _ = next(anchor_iter)
            anchor_obs = anchor_obs.to(device, non_blocking=True)

            with torch.amp.autocast(device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                # Teacher CE on corrections — frozen BN (OOD crisis states must
                # not poison running stats; eager module so the freeze takes
                # effect under torch.compile).
                with frozen_bn(base_model):
                    t_out = base_model(corr['obs'][idx])
                t_logits = t_out[0] if isinstance(t_out, tuple) else t_out
                teacher_loss = soft_correction_loss(
                    t_logits, corr['tgt_idx'][idx], corr['tgt_prob'][idx],
                    anchor_weight=(corr['weight'][idx] if args.weighted else None),
                    target_temperature=args.target_temperature)
                # Anchor KL on broad states — separate forward (BN sees only the
                # in-distribution batch), frozen center under no_grad.
                s_out = model(anchor_obs)
                s_logits = s_out[0] if isinstance(s_out, tuple) else s_out
                with torch.no_grad():
                    f_out = frozen(anchor_obs)
                    f_logits = f_out[0] if isinstance(f_out, tuple) else f_out
                kl = anchor_kl(f_logits, s_logits)
                loss = teacher_loss + args.beta * kl

            audit = (args.grad_audit_every > 0
                     and global_step % args.grad_audit_every == 0
                     and global_step > 0)
            if audit:
                params = [q for q in model.parameters() if q.requires_grad]

                def _gnorm():
                    return torch.cat(
                        [(q.grad.detach().flatten() if q.grad is not None
                          else torch.zeros(q.numel(), device=device))
                         for q in params]).norm().item()
                optimizer.zero_grad(set_to_none=True)
                teacher_loss.backward(retain_graph=True)
                g_t = _gnorm()
                optimizer.zero_grad(set_to_none=True)
                kl.backward(retain_graph=True)
                g_a = _gnorm()
                share = args.beta * g_a / max(g_t, 1e-9)
                print(f"  [grad step {global_step}] |g_teacher|={g_t:.4f} "
                      f"|g_anchor|={g_a:.4f} β|g_a|/|g_t|={share:.3f}",
                      flush=True)

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
            tl_sum += float(teacher_loss.detach())
            kl_sum += float(kl.detach())
            n_steps += 1
            global_step += 1

        scheduler.step()
        vce, vkl = validate_drift(base_model, frozen, val_loader, device,
                                  amp_dtype)
        mr, ce = corrections_metrics(base_model, corr, args.teacher_batch_size,
                                     device, amp_dtype)
        line = (f"Epoch {epoch+1}/{args.epochs} "
                f"teacher={tl_sum/n_steps:.4f} anchorKL={kl_sum/n_steps:.5f} | "
                f"val: V13ce={vce:.4f} KLvsFrozen={vkl:.5f} | "
                f"corr match={mr:.3f} ce={ce:.3f}")
        if corr_held is not None:
            mrh, ceh = corrections_metrics(base_model, corr_held,
                                           args.teacher_batch_size, device,
                                           amp_dtype)
            line += f" | HELD match={mrh:.3f} ce={ceh:.3f}"
        print(line + f"  [{time.time()-et:.0f}s lr={optimizer.param_groups[0]['lr']:.2e}]",
              flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            ck = {'epoch': epoch, 'model': model.state_dict(),
                  'args': vars(args), 'policy_only': True,
                  'val_loss': vce, 'anchor_kl': vkl}
            path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pt')
            torch.save(ck, path)
            print(f"  saved {path}", flush=True)
            if args.copy_to:
                copy_dir = os.path.dirname(args.copy_to) or '.'
                run_prefix = (os.path.basename(args.save_dir.rstrip('/'))
                              or 'run')
                shutil.copy2(path, os.path.join(
                    copy_dir, f'{run_prefix}_epoch_{epoch+1}.pt'))

    print(f"\n=== Done in {(time.time()-t_total)/60:.1f}m ===", flush=True)


if __name__ == '__main__':
    main()
