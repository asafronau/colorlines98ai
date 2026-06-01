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
    listwise_margin_loss,
    preflight_metrics as aux_preflight_metrics,
    DEFAULT_MARGIN as AUX_DEFAULT_MARGIN,
    DEFAULT_TOP1_WEIGHT as AUX_DEFAULT_TOP1_WEIGHT,
    DEFAULT_OTHER_WEIGHT as AUX_DEFAULT_OTHER_WEIGHT,
)
from alphatrain.dataset import NUM_CHANNELS, TensorDatasetGPU
from alphatrain.model import AlphaTrainNet, count_parameters


def cross_entropy_soft(logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def distillation_loss(logits, soft_targets, blend_alpha=1.0,
                      target_temperature=1.0):
    """Cross-entropy on a visit-distribution target.

    target_temperature=1.0 (default): targets used as-stored.
    target_temperature<1.0: sharpen targets via `target**(1/T)` renormalized.
        Forces the model to commit to high-visit moves. Engaged sharp_50's
        +57% lift over baseline.
    blend_alpha<1.0: convex blend of soft target CE and hard CE on argmax.
        Rarely used; legacy from pre-sharpening experiments.
    """
    if target_temperature != 1.0:
        sharp = soft_targets.pow(1.0 / target_temperature)
        sharp = sharp / sharp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        soft_targets = sharp
    log_probs = F.log_softmax(logits, dim=-1)
    soft = -(soft_targets * log_probs).sum(dim=-1).mean()
    if blend_alpha >= 1.0:
        return soft
    argmax_idx = soft_targets.argmax(dim=-1)
    hard = F.cross_entropy(logits, argmax_idx)
    return blend_alpha * soft + (1.0 - blend_alpha) * hard


def _aux_lambda_schedule(step_in_epoch, steps_per_epoch, epoch,
                          target_lambda, warmup_epochs):
    """Linear ramp 0 → target_lambda over `warmup_epochs` epochs."""
    if warmup_epochs <= 0:
        return target_lambda
    global_step = epoch * steps_per_epoch + step_in_epoch
    warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))
    return target_lambda * min(1.0, global_step / warmup_steps)


def train_epoch(model, loader, optimizer, device, scaler, amp_dtype,
                 log_interval=100, blend_alpha=1.0, target_temperature=1.0,
                 aux=None, epoch=0):
    """One epoch. Optionally adds the listwise margin aux loss.

    `aux`, when not None, is a dict with:
        obs, winner_idx, top1_idx, loser_idx, loser_mask  — all on device
        batch_size, target_lambda, margin, top1_weight, other_weight,
        warmup_epochs, preflight_every, abort_flip_rate (optional float)
        ptr: persistent int across batches (passed as a mutable list).
    """
    model.train(True)
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
                target_temperature=target_temperature)

            if aux is not None:
                lam = _aux_lambda_schedule(
                    bi, steps_per_epoch, epoch,
                    aux['target_lambda'], aux['warmup_epochs'])
                if lam > 0.0:
                    aux_obs, aux_idx = _next_aux_batch(aux)
                    aux_out = model(aux_obs)
                    aux_logits = (aux_out[0] if isinstance(aux_out, tuple)
                                  else aux_out)
                    aux_loss = listwise_margin_loss(
                        aux_logits,
                        aux['winner_idx'][aux_idx],
                        aux['top1_idx'][aux_idx],
                        aux['loser_idx'][aux_idx],
                        aux['loser_mask'][aux_idx],
                        margin=aux['margin'],
                        top1_weight=aux['top1_weight'],
                        other_weight=aux['other_weight'])
                    loss = main_loss + lam * aux_loss
                    total_aux += float(aux_loss.detach())
                    aux_steps += 1
                else:
                    loss = main_loss
            else:
                loss = main_loss

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
            _run_preflight(model, aux, device, amp_dtype, epoch, bi + 1,
                           steps_per_epoch)

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
def _run_preflight(model, aux, device, amp_dtype, epoch, step_in_epoch,
                    steps_per_epoch):
    """Run preflight metrics on the full aux corpus and print."""
    was_training = model.training
    model.train(False)
    use_amp = amp_dtype != torch.float32
    N = aux['obs'].size(0)
    bs = max(1, aux['batch_size'])
    chunks = []
    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
        for i in range(0, N, bs):
            out = model(aux['obs'][i:i+bs])
            chunks.append(out[0] if isinstance(out, tuple) else out)
    logits = torch.cat(chunks, dim=0).float()
    m = aux_preflight_metrics(
        logits, aux['winner_idx'], aux['top1_idx'],
        aux['loser_idx'], aux['loser_mask'], margin=aux['margin'])
    print(f"  [preflight ep{epoch+1} step {step_in_epoch}/{steps_per_epoch}] "
          f"top1_flip={m['stored_top1_flip_rate']:.4f} "
          f"clean_margin@{aux['margin']}={m['all_clean_loser_margin_rate']:.4f}",
          flush=True)
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
    args = p.parse_args()

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
        optimizer, T_max=args.epochs - args.warmup_epochs,
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
                                  aux=aux, epoch=epoch)
        vl = validate(model, val_loader, device, amp_dtype=amp_dtype)
        scheduler.step()

        print(f"  Train: loss={tl:.4f}"
              + (f"  aux_loss={aux_tl:.4f}" if aux is not None else ""),
              flush=True)
        print(f"  V12 val: loss={vl:.4f} [{time.time()-et:.0f}s]",
              flush=True)
        if aux is not None:
            _run_preflight(model, aux, device, amp_dtype,
                           epoch=epoch, step_in_epoch=len(train_loader),
                           steps_per_epoch=len(train_loader))

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
