"""AlphaTrain V-corpus distillation with optional target sharpening.

Trains a policy network on MCTS visit-distribution targets (V11/V12/V13/...).
Color permutation + dihedral augmentation are on by default (color was the
+4% lift in Path B v1's ablation; HISTORY 143-144). Target sharpening via
`--target-temperature` is the load-bearing knob that produced sharp_50's
+57% gameplay lift over baseline (HISTORY 153-156).

The legacy name "train_path_b" comes from the Path B v1 oracle experiment
(soft-KL auxiliary loss). That branch is closed (HISTORY 145, 151-152).
This script is now V-corpus distillation only — no oracle path, no λ tuning,
no auxiliary corpus. The training recipe that produces pillar3a / pillar4a /
pillar5a is just this.

Usage (Colab, ~12h for 17 epochs on G4 / L4):
    python -m alphatrain.train_path_b \\
        --tensor-file alphatrain/data/v13_pillar3a.pt \\
        --amp --compile \\
        --resume alphatrain/data/pillar3a.pt --warm-start \\
        --epochs 17 --batch-size 32768 --lr 3e-4 --warmup-epochs 1 \\
        --target-temperature 0.5 \\
        --copy-to /content/drive/MyDrive/alphatrain/pillar4a_best.pt \\
        --save-dir /content/checkpoints/pillar4a

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

from alphatrain.dataset import TensorDatasetGPU
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


def train_epoch(model, loader, optimizer, device, scaler, amp_dtype,
                 log_interval=100, blend_alpha=1.0, target_temperature=1.0):
    """One epoch. Single forward per V12 batch."""
    model.train(True)
    use_amp = amp_dtype != torch.float32
    total_loss = 0.0
    n = 0
    t0 = time.time()

    for bi, batch in enumerate(loader):
        obs, pol_tgt = batch
        obs = obs.to(device, non_blocking=True)
        pol_tgt = pol_tgt.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(obs)
            logits = out[0] if isinstance(out, tuple) else out
            loss = distillation_loss(
                logits, pol_tgt,
                blend_alpha=blend_alpha,
                target_temperature=target_temperature)

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
            print(f"  [{bi+1}/{len(loader)}] "
                  f"loss={total_loss/n:.4f} "
                  f"{sps:.0f} s/s ETA {eta/60:.0f}m", flush=True)

    return total_loss / n


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

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\n=== Training {args.epochs} epochs ===", flush=True)
    t_total = time.time()

    for epoch in range(start_epoch, args.epochs):
        et = time.time()
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.2e})", flush=True)

        tl = train_epoch(model, train_loader, optimizer, device,
                          scaler, amp_dtype,
                          blend_alpha=args.blend_alpha,
                          target_temperature=args.target_temperature)
        vl = validate(model, val_loader, device, amp_dtype=amp_dtype)
        scheduler.step()

        print(f"  Train: loss={tl:.4f}", flush=True)
        print(f"  V12 val: loss={vl:.4f} [{time.time()-et:.0f}s]",
              flush=True)

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
