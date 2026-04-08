"""AlphaTrain training script.

Usage:
    python -m alphatrain.train --tensor-file data/alphatrain_v1.pt --gpu-data --amp
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from alphatrain.model import AlphaTrainNet, count_parameters
from alphatrain.dataset import TensorDatasetGPU, SelfPlayDataset


def cross_entropy_soft(logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def train_epoch(model, loader, optimizer, device, max_score=500.0,
                num_value_bins=64, scaler=None, pairwise=False,
                scalar_value=False, selfplay=False, val_weight=1.0,
                rank_weight=1.0, anchor_weight=0.0, log_interval=100):
    model.train()
    total_loss = 0
    total_pol = 0
    total_val = 0
    total_rank = 0
    total_anchor = 0
    n = 0
    t0 = time.time()
    use_amp = scaler is not None
    # Precompute bin centers for decoding two-hot targets to scalar
    if anchor_weight > 0 and scalar_value:
        anchor_bins = torch.linspace(0, max_score, num_value_bins, device=device)
    for bi, batch in enumerate(loader):
        if pairwise:
            obs, pol_tgt, val_tgt, good_obs, bad_obs, margin = batch
        else:
            obs, pol_tgt, val_tgt = batch
        obs = obs.to(device)
        pol_tgt = pol_tgt.to(device)
        val_tgt = val_tgt.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            pol_logits, val_logits = model(obs)
            pol_loss = cross_entropy_soft(pol_logits, pol_tgt)

            if selfplay:
                # Self-play: direct MSE on scalar value targets
                v_pred = model.predict_value(val_logits, max_val=max_score)
                val_loss = F.mse_loss(v_pred, val_tgt)
                loss = pol_loss + val_weight * val_loss
                anchor_loss = torch.tensor(0.0, device=device)
            elif scalar_value:
                # Legacy scalar head with anchor loss
                val_loss = torch.tensor(0.0, device=device)
                loss = pol_loss
            else:
                # Categorical value head
                val_loss = cross_entropy_soft(val_logits, val_tgt)
                loss = pol_loss + val_weight * val_loss

            # Anchor loss: MSE between sigmoid-clamped prediction and TD target
            if not selfplay and anchor_weight > 0 and scalar_value:
                v_pred = model.predict_value(val_logits, max_val=max_score)
                true_scalar = (val_tgt * anchor_bins).sum(dim=-1)
                anchor_loss = F.mse_loss(v_pred, true_scalar)
                loss = loss + anchor_weight * anchor_loss
            elif not selfplay:
                anchor_loss = torch.tensor(0.0, device=device)

            if pairwise:
                good_obs = good_obs.to(device)
                bad_obs = bad_obs.to(device)
                margin = margin.to(device)
                # Single forward pass for both afterstates (3→2 total passes)
                pair_obs = torch.cat([good_obs, bad_obs], dim=0)
                _, pair_val = model(pair_obs)
                good_val, bad_val = pair_val.chunk(2, dim=0)
                v_good = model.predict_value(good_val, max_val=max_score)
                v_bad = model.predict_value(bad_val, max_val=max_score)
                # Ranking: V(good) must exceed V(bad) by scaled margin
                margin_scaled = margin * (5.0 / (margin.mean() + 1e-8))
                rank_loss = F.relu(margin_scaled - (v_good - v_bad)).mean()
                loss = loss + rank_weight * rank_loss

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
        total_pol += pol_loss.item()
        total_val += val_loss.item()
        total_anchor += anchor_loss.item()
        if pairwise:
            total_rank += rank_loss.item()
        n += 1

        if (bi + 1) % log_interval == 0:
            elapsed = time.time() - t0
            sps = (bi + 1) * loader.batch_size / elapsed
            eta = (len(loader) - bi - 1) * loader.batch_size / max(sps, 1)
            rank_str = f" rank={total_rank/n:.4f}" if pairwise else ""
            anchor_str = f" anchor={total_anchor/n:.4f}" if anchor_weight > 0 else ""
            print(f"  [{bi+1}/{len(loader)}] "
                  f"loss={total_loss/n:.4f} "
                  f"(pol={total_pol/n:.4f} val={total_val/n:.4f}"
                  f"{rank_str}{anchor_str}) "
                  f"{sps:.0f} s/s ETA {eta/60:.0f}m", flush=True)

    return total_loss / n, total_pol / n, total_val / n


@torch.no_grad()
def validate(model, loader, device, max_score=30000.0, num_value_bins=64,
             use_amp=False, scalar_value=False, selfplay=False):
    model.eval()
    total_loss = 0
    total_pol = 0
    total_val = 0
    total_mae = 0
    n = 0
    # Bin centers for decoding two-hot to scalar (legacy)
    if not selfplay:
        bins = torch.linspace(0, max_score, num_value_bins, device=device)

    for obs, pol_tgt, val_tgt in loader:
        obs = obs.to(device)
        pol_tgt = pol_tgt.to(device)
        val_tgt = val_tgt.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            pol_logits, val_logits = model(obs)
            pol_loss = cross_entropy_soft(pol_logits, pol_tgt)
            if selfplay:
                pred = model.predict_value(val_logits, max_val=max_score)
                val_loss = F.mse_loss(pred, val_tgt)
            elif scalar_value:
                val_loss = torch.tensor(0.0, device=device)
            else:
                val_loss = cross_entropy_soft(val_logits, val_tgt)

        # MAE
        pred = model.predict_value(val_logits, max_val=max_score)
        if selfplay:
            true = val_tgt
        else:
            true = (val_tgt * bins).sum(dim=-1)
        mae = (pred - true).abs().mean()

        total_loss += (pol_loss + val_loss).item()
        total_pol += pol_loss.item()
        total_val += val_loss.item()
        total_mae += mae.item()
        n += 1

    return (total_loss / n, total_pol / n, total_val / n, total_mae / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor-file', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--val-weight', type=float, default=1.0,
                   help='Weight for value CE loss (default 1.0)')
    p.add_argument('--rank-weight', type=float, default=1.0,
                   help='Weight for pairwise ranking loss (default 1.0)')
    p.add_argument('--anchor-weight', type=float, default=0.0,
                   help='Weight for anchor MSE loss (scalar head only, default 0.0)')
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--value-bins', type=int, default=64,
                   help='Value head output size (1=scalar for pure ranking, 64=categorical)')
    p.add_argument('--value-channels', type=int, default=32,
                   help='Value head conv channels (default 32)')
    p.add_argument('--value-hidden', type=int, default=512,
                   help='Value head FC hidden size (default 512)')
    p.add_argument('--value-dropout', type=float, default=0.0,
                   help='Value head dropout rate (default 0.0)')
    p.add_argument('--val-split', type=float, default=0.05)
    p.add_argument('--gpu-data', action='store_true')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--compile', action='store_true',
                   help='Use torch.compile for faster forward/backward')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--warm-start', action='store_true',
                   help='Load weights from --resume but reset optimizer/scheduler')
    p.add_argument('--freeze-backbone', action='store_true',
                   help='Freeze stem + blocks + policy head, train only value head')
    p.add_argument('--save-dir', default='checkpoints/alphatrain')
    p.add_argument('--copy-to', type=str, default=None)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--trap-fraction', type=float, default=0.0,
                   help='Fraction of batch to replace with trap boards (value=0)')
    p.add_argument('--endgame-fraction', type=float, default=0.0,
                   help='Fraction of batch to replace with endgame positions')
    p.add_argument('--endgame-threshold', type=int, default=100,
                   help='Turns remaining threshold for endgame positions')
    p.add_argument('--adversarial-ranking', action='store_true',
                   help='Use top-1 vs random move pairs instead of top-1 vs top-5')
    args = p.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    gpu_data = args.gpu_data and device.type in ('cuda', 'mps')
    if not gpu_data:
        raise NotImplementedError("Use --gpu-data")

    # Detect data format
    probe = torch.load(args.tensor_file, weights_only=False, map_location='cpu')
    selfplay = (probe.get('format') == 'selfplay')
    del probe

    if selfplay:
        dataset = SelfPlayDataset(args.tensor_file, augment=True, device=str(device))
    else:
        dataset = TensorDatasetGPU(args.tensor_file, augment=True, device=str(device),
                                   trap_fraction=args.trap_fraction,
                                   endgame_fraction=args.endgame_fraction,
                                   endgame_threshold=args.endgame_threshold,
                                   adversarial_ranking=args.adversarial_ranking)

    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    pairwise = dataset.has_pairs
    collate_fn = dataset.collate_pairwise if pairwise else dataset.collate
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=0, collate_fn=dataset.collate)  # val always standard

    max_score = dataset.max_score
    pairwise_str = ", pairwise=ON" if pairwise else ""
    print(f"Train: {n_train:,}, Val: {n_val:,}, max_score: {max_score:.0f}"
          f"{pairwise_str}", flush=True)

    model = AlphaTrainNet(num_blocks=args.num_blocks, channels=args.channels,
                          num_value_bins=args.value_bins,
                          value_channels=args.value_channels,
                          value_hidden=args.value_hidden,
                          value_dropout=args.value_dropout).to(device)
    scalar_value = (args.value_bins == 1)
    n_params = count_parameters(model)
    # channels_last gives better perf for small spatial dims on CUDA
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    print(f"Model: {args.num_blocks}b x {args.channels}ch, "
          f"{n_params:,} params, {model.in_channels}ch input", flush=True)

    # Load weights BEFORE torch.compile (compile wraps state dict keys)
    start_epoch = 0
    best_val = float('inf')

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt['model']
        # Strip torch.compile prefix if present in checkpoint
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        # Filter out size-mismatched keys (e.g., value_fc2: 64 bins → 1 scalar)
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

    # Freeze backbone + policy head if requested (train value head only)
    if args.freeze_backbone:
        frozen, trainable = 0, 0
        value_prefixes = ('value_conv', 'value_bn', 'value_fc1', 'value_fc2')
        for name, param in model.named_parameters():
            if name.startswith(value_prefixes):
                param.requires_grad_(True)
                trainable += param.numel()
            else:
                param.requires_grad_(False)
                frozen += param.numel()
        print(f"Frozen backbone: {frozen:,} params frozen, "
              f"{trainable:,} trainable (value head only)", flush=True)

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

        tl, tp, tv = train_epoch(model, train_loader, optimizer, device,
                                  max_score=max_score,
                                  num_value_bins=dataset.num_value_bins,
                                  scaler=scaler,
                                  pairwise=pairwise, scalar_value=scalar_value,
                                  selfplay=selfplay,
                                  val_weight=args.val_weight,
                                  rank_weight=args.rank_weight,
                                  anchor_weight=args.anchor_weight)
        vl, vp, vv, vm = validate(model, val_loader, device,
                                   max_score=max_score,
                                   num_value_bins=dataset.num_value_bins,
                                   use_amp=use_amp,
                                   scalar_value=scalar_value,
                                   selfplay=selfplay)
        scheduler.step()

        print(f"  Train: loss={tl:.4f} (pol={tp:.4f} val={tv:.4f})", flush=True)
        print(f"  Val:   loss={vl:.4f} (pol={vp:.4f} val={vv:.4f} MAE={vm:.0f}) "
              f"[{time.time()-et:.0f}s]", flush=True)

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': vl, 'val_mae': vm,
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
