"""AlphaTrain training script (policy-only).

Usage:
    python -m alphatrain.train --tensor-file data/selfplay.pt --amp --compile
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from alphatrain.model import AlphaTrainNet, count_parameters
from alphatrain.dataset import TensorDatasetGPU


def cross_entropy_soft(logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def train_epoch(model, loader, optimizer, device, scaler=None, log_interval=100):
    model.train()
    total_loss = 0
    n = 0
    t0 = time.time()
    use_amp = scaler is not None

    for bi, batch in enumerate(loader):
        obs, pol_tgt = batch
        obs = obs.to(device)
        pol_tgt = pol_tgt.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            pol_logits = model(obs)
            loss = cross_entropy_soft(pol_logits, pol_tgt)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
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
def validate(model, loader, device, use_amp=False):
    model.eval()
    total_loss = 0
    n = 0

    for obs, pol_tgt in loader:
        obs = obs.to(device)
        pol_tgt = pol_tgt.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            pol_logits = model(obs)
            loss = cross_entropy_soft(pol_logits, pol_tgt)

        total_loss += loss.item()
        n += 1

    return total_loss / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor-file', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--val-split', type=float, default=0.05)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--compile', action='store_true',
                   help='Use torch.compile for faster forward/backward')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--warm-start', action='store_true',
                   help='Load weights from --resume but reset optimizer/scheduler')
    p.add_argument('--save-dir', default='checkpoints/alphatrain')
    p.add_argument('--copy-to', type=str, default=None)
    p.add_argument('--num-workers', type=int, default=8)
    args = p.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    dataset = TensorDatasetGPU(args.tensor_file, augment=True, device=str(device))

    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=dataset.collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=0, collate_fn=dataset.collate)

    max_score = dataset.max_score
    print(f"Train: {n_train:,}, Val: {n_val:,}, max_score: {max_score:.0f}",
          flush=True)

    model = AlphaTrainNet(num_blocks=args.num_blocks, channels=args.channels).to(device)
    n_params = count_parameters(model)
    # channels_last gives better perf for small spatial dims on CUDA
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    print(f"Model: {args.num_blocks}b x {args.channels}ch, "
          f"{n_params:,} params, {model.in_channels}ch input", flush=True)

    # Load weights BEFORE torch.compile (compile wraps state dict keys)
    start_epoch = 0
    best_val = float('inf')

    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(
            f"--resume file not found: {args.resume}\n"
            f"  Training would start from scratch -- refusing to continue.\n"
            f"  Check the path or remove --resume to train from scratch intentionally.")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt['model']
        # Strip torch.compile prefix if present in checkpoint
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        # Filter out size-mismatched keys (e.g., value head keys from old checkpoint)
        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        missing, _ = model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}", flush=True)
        if missing:
            print(f"  Randomly initialized: {[k for k in missing if k not in skipped]}",
                  flush=True)
        if not args.warm_start and 'optimizer' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            best_val = ckpt.get('best_val_loss', float('inf'))
            print(f"Resumed weights from epoch {start_epoch - 1}", flush=True)
        else:
            print(f"Warm start: loaded weights from epoch {ckpt.get('epoch', '?')}, "
                  f"fresh optimizer/scheduler", flush=True)

    # torch.compile AFTER loading weights
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("torch.compile enabled", flush=True)

    # Only pass trainable params to optimizer
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(train_params, lr=args.lr,
                                   weight_decay=args.weight_decay)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=args.warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched],
        milestones=[args.warmup_epochs])

    # Restore optimizer/scheduler state for full resume (not warm start)
    if args.resume and os.path.exists(args.resume) and not args.warm_start:
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        print(f"Resumed from epoch {start_epoch}", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("AMP enabled", flush=True)

    print(f"\n=== Training {args.epochs} epochs ===", flush=True)
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        et = time.time()
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.2e})", flush=True)

        tl = train_epoch(model, train_loader, optimizer, device, scaler=scaler)
        vl = validate(model, val_loader, device, use_amp=use_amp)
        scheduler.step()

        print(f"  Train: loss={tl:.4f}", flush=True)
        print(f"  Val:   loss={vl:.4f} [{time.time()-et:.0f}s]", flush=True)

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': vl,
            'best_val_loss': min(best_val, vl),
            'max_score': max_score,
            'args': vars(args),
        }
        latest_path = os.path.join(args.save_dir, 'latest.pt')
        torch.save(ckpt, latest_path)
        # Save per-epoch checkpoint (never overwritten)
        epoch_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pt')
        torch.save(ckpt, epoch_path)
        if args.copy_to:
            import shutil
            copy_dir = os.path.dirname(args.copy_to) or '.'
            # Always save latest to Drive for crash recovery
            shutil.copy2(latest_path, os.path.join(copy_dir, 'alphatrain_td_latest.pt'))
            # Save per-epoch to Drive too
            shutil.copy2(epoch_path, os.path.join(copy_dir, f'alphatrain_td_epoch_{epoch+1}.pt'))
        if vl < best_val:
            best_val = vl
            best_path = os.path.join(args.save_dir, 'best.pt')
            torch.save(ckpt, best_path)
            print(f"  ** New best (val_loss={vl:.4f}) **", flush=True)
            if args.copy_to:
                shutil.copy2(best_path, args.copy_to)
                print(f"  Copied best to {args.copy_to}", flush=True)

    print(f"\n=== Done: {args.epochs} epochs in {(time.time()-t0)/3600:.1f}h ===",
          flush=True)
    print(f"Best val loss: {best_val:.4f}", flush=True)


if __name__ == '__main__':
    main()
