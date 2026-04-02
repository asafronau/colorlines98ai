"""Train standalone ValueNet (PolicyNet frozen, no shared backbone).

The ValueNet learns to predict game outcomes from board positions.
PolicyNet is NOT loaded or modified — this trains value only.

Usage:
    python -m alphatrain.train_value \
        --tensor-file alphatrain/data/selfplay_iter1.pt \
        --gpu-data --epochs 20 --lr 1e-3 \
        --num-blocks 6 --channels 128
"""

import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from alphatrain.model import ValueNet, count_parameters
from alphatrain.dataset import SelfPlayDataset


def train_epoch(model, loader, optimizer, device, max_score=500.0,
                scaler=None, log_interval=100):
    model.train()
    total_loss = 0
    total_mae = 0
    n = 0
    t0 = time.time()
    use_amp = scaler is not None

    for bi, batch in enumerate(loader):
        obs, _pol, val_tgt = batch
        obs = obs.to(device)
        val_tgt = val_tgt.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            val_logits = model(obs)
            val_pred = model.predict_value(val_logits, max_val=max_score)
            loss = F.mse_loss(val_pred, val_tgt)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            mae = (val_pred - val_tgt).abs().mean().item()

        total_loss += loss.item()
        total_mae += mae
        n += 1

        if (bi + 1) % log_interval == 0:
            elapsed = time.time() - t0
            sps = (bi + 1) * loader.batch_size / elapsed
            eta = (len(loader) - bi - 1) * loader.batch_size / max(sps, 1)
            print(f"  [{bi+1}/{len(loader)}] "
                  f"mse={total_loss/n:.2f} mae={total_mae/n:.1f} "
                  f"{sps:.0f} s/s ETA {eta/60:.0f}m", flush=True)

    return total_loss / n, total_mae / n


@torch.no_grad()
def validate(model, loader, device, max_score=500.0, use_amp=False):
    model.eval()
    total_loss = 0
    total_mae = 0
    n = 0

    for obs, _pol, val_tgt in loader:
        obs = obs.to(device)
        val_tgt = val_tgt.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            val_logits = model(obs)
            val_pred = model.predict_value(val_logits, max_val=max_score)
            loss = F.mse_loss(val_pred, val_tgt)

        mae = (val_pred - val_tgt).abs().mean().item()
        total_loss += loss.item()
        total_mae += mae
        n += 1

    return total_loss / n, total_mae / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor-file', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-blocks', type=int, default=6)
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--val-split', type=float, default=0.05)
    p.add_argument('--gpu-data', action='store_true')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--resume', type=str, default=None,
                   help='Resume from ValueNet checkpoint')
    p.add_argument('--save-dir', default='checkpoints/value_net')
    p.add_argument('--copy-to', type=str, default=None)
    args = p.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    if not (args.gpu_data and device.type in ('cuda', 'mps')):
        raise NotImplementedError("Use --gpu-data with CUDA or MPS")

    dataset = SelfPlayDataset(args.tensor_file, augment=True, device=str(device))
    max_score = dataset.max_score

    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=dataset.collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=0, collate_fn=dataset.collate)

    print(f"Train: {n_train:,}, Val: {n_val:,}, max_score: {max_score:.0f}",
          flush=True)

    model = ValueNet(num_blocks=args.num_blocks, channels=args.channels,
                     num_value_bins=1).to(device)
    n_params = count_parameters(model)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    print(f"ValueNet: {args.num_blocks}b x {args.channels}ch, "
          f"{n_params:,} params", flush=True)

    start_epoch = 0
    best_val = float('inf')

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt['model']
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}", flush=True)

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("torch.compile enabled", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
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

    os.makedirs(args.save_dir, exist_ok=True)
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f"\n=== Training ValueNet: {args.epochs} epochs ===", flush=True)
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        et = time.time()
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.2e})", flush=True)

        tl, tm = train_epoch(model, train_loader, optimizer, device,
                              max_score=max_score, scaler=scaler)
        vl, vm = validate(model, val_loader, device,
                           max_score=max_score, use_amp=use_amp)
        scheduler.step()

        print(f"  Train: mse={tl:.2f} mae={tm:.1f}", flush=True)
        print(f"  Val:   mse={vl:.2f} mae={vm:.1f} [{time.time()-et:.0f}s]",
              flush=True)

        ckpt_data = {
            'epoch': epoch,
            'model': model.state_dict(),
            'val_loss': vl, 'val_mae': vm,
            'best_val_loss': min(best_val, vl),
            'max_score': max_score,
            'model_type': 'value_net',
            'args': vars(args),
        }
        latest_path = os.path.join(args.save_dir, 'latest.pt')
        torch.save(ckpt_data, latest_path)
        epoch_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pt')
        torch.save(ckpt_data, epoch_path)

        if args.copy_to:
            import shutil
            copy_dir = os.path.dirname(args.copy_to) or '.'
            shutil.copy2(latest_path,
                         os.path.join(copy_dir, 'value_net_latest.pt'))
            shutil.copy2(epoch_path,
                         os.path.join(copy_dir, f'value_net_epoch_{epoch+1}.pt'))

        if vl < best_val:
            best_val = vl
            best_path = os.path.join(args.save_dir, 'best.pt')
            torch.save(ckpt_data, best_path)
            print(f"  ** New best (val_mse={vl:.2f}, mae={vm:.1f}) **",
                  flush=True)
            if args.copy_to:
                shutil.copy2(best_path, args.copy_to)
                print(f"  Copied best to {args.copy_to}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== Done: {args.epochs} epochs in {elapsed/3600:.1f}h ===",
          flush=True)
    print(f"Best val MSE: {best_val:.2f}", flush=True)


if __name__ == '__main__':
    main()
