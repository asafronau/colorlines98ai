"""Hybrid training: interleaved expert (pairwise ranking) + self-play (MSE value).

Expert batches: pol_CE + 0.001 * (rank_loss + anchor_MSE)
Self-play batches: pol_CE + 0.001 * MSE(value, TD_return)

Uses two dataloaders — one for each data source — interleaved within each epoch.
This preserves the pairwise ranking signal that made Pillar 2f successful while
absorbing self-play value targets for positions the model actually encounters.

GPU Memory considerations (H100 80GB):
    Both datasets are GPU-resident: expert ~5 GB (sparse boards + pairs),
    selfplay ~11 GB (pre-computed obs 346K×18×9×9 + dense policy 346K×6561).
    Total dataset footprint ~65 GB, leaving ~15 GB for model + activations.

    Each training step has 3 forward passes (expert main bs=N, expert pairs
    bs=2N, selfplay bs=N). To avoid OOM, we use split backward passes with
    gradient accumulation: expert backward runs and frees activations BEFORE
    selfplay forward starts. torch.cuda.empty_cache() between passes prevents
    fragmentation from torch.compile's cached allocations.

    Tested batch sizes on H100-80GB: bs=2048 OOM, bs=1024 fits.
    On A100-40GB: use bs=512. On A100-80GB: use bs=1024.

Usage:
    python -m alphatrain.train_hybrid \
        --expert-file alphatrain/data/alphatrain_pairwise.pt \
        --selfplay-file alphatrain/data/selfplay_iter2.pt \
        --resume alphatrain/data/pillar2f_best.pt --warm-start \
        --epochs 15 --batch-size 1024 --lr 5e-5 --warmup-epochs 2 \
        --val-weight 0.001 --rank-weight 1.0 --anchor-weight 0.001
"""

import os
import time
import argparse
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from alphatrain.model import AlphaTrainNet, count_parameters
from alphatrain.dataset import TensorDatasetGPU, SelfPlayDataset


def cross_entropy_soft(logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def train_epoch_hybrid(model, expert_loader, selfplay_loader, optimizer, device,
                       max_score=500.0, num_value_bins=64, scaler=None,
                       val_weight=0.001, rank_weight=1.0, anchor_weight=0.001,
                       log_interval=50):
    """One epoch of interleaved expert + self-play training.

    Expert is the longer loader (1.31M states). Self-play cycles to match.
    Each step processes one expert batch and one self-play batch, averages losses.
    """
    model.train()
    total_loss = 0
    total_pol_e = 0
    total_rank = 0
    total_anchor = 0
    total_pol_s = 0
    total_val_s = 0
    n = 0
    t0 = time.time()
    use_amp = scaler is not None

    # Precompute bin centers for decoding two-hot to scalar (expert anchor loss)
    anchor_bins = torch.linspace(0, max_score, num_value_bins, device=device)

    # Expert is larger — cycle selfplay to match
    selfplay_iter = itertools.cycle(selfplay_loader)

    for bi, expert_batch in enumerate(expert_loader):
        selfplay_batch = next(selfplay_iter)

        # ── Expert: pol_CE + ranking + anchor ──
        obs_e, pol_e, val_e, good_obs, bad_obs, margin = expert_batch
        obs_e = obs_e.to(device)
        pol_e = pol_e.to(device)
        val_e = val_e.to(device)
        good_obs = good_obs.to(device)
        bad_obs = bad_obs.to(device)
        margin = margin.to(device)

        # ── Self-play: pol_CE + MSE value ──
        obs_s, pol_s, val_s = selfplay_batch
        obs_s = obs_s.to(device)
        pol_s = pol_s.to(device)
        val_s = val_s.to(device)

        # Two separate backward passes to halve peak activation memory.
        # Gradient accumulation makes this mathematically identical to
        # a single backward on (loss_expert + loss_selfplay) / 2.
        optimizer.zero_grad(set_to_none=True)

        # ── Pass 1: Expert (pol_CE + ranking + anchor) ──
        with torch.amp.autocast('cuda', enabled=use_amp):
            pol_logits_e, val_logits_e = model(obs_e)
            pol_loss_e = cross_entropy_soft(pol_logits_e, pol_e)

            v_pred_e = model.predict_value(val_logits_e, max_val=max_score)
            true_scalar = (val_e * anchor_bins).sum(dim=-1)
            anchor_loss = F.mse_loss(v_pred_e, true_scalar)

            pair_obs = torch.cat([good_obs, bad_obs], dim=0)
            _, pair_val = model(pair_obs)
            good_val, bad_val = pair_val.chunk(2, dim=0)
            v_good = model.predict_value(good_val, max_val=max_score)
            v_bad = model.predict_value(bad_val, max_val=max_score)
            margin_scaled = margin * (5.0 / (margin.mean() + 1e-8))
            rank_loss = F.relu(margin_scaled - (v_good - v_bad)).mean()

            loss_expert = (pol_loss_e
                           + val_weight * anchor_loss
                           + rank_weight * rank_loss) / 2.0  # /2 for averaging

        if use_amp:
            scaler.scale(loss_expert).backward()
        else:
            loss_expert.backward()

        # Free expert activations and defragment before selfplay forward
        del obs_e, pol_e, val_e, good_obs, bad_obs, margin
        del pol_logits_e, val_logits_e, pair_obs, pair_val
        torch.cuda.empty_cache()

        # ── Pass 2: Self-play (pol_CE + MSE value) ──
        with torch.amp.autocast('cuda', enabled=use_amp):
            pol_logits_s, val_logits_s = model(obs_s)
            pol_loss_s = cross_entropy_soft(pol_logits_s, pol_s)
            v_pred_s = model.predict_value(val_logits_s, max_val=max_score)
            val_loss_s = F.mse_loss(v_pred_s, val_s)

            loss_selfplay = (pol_loss_s + val_weight * val_loss_s) / 2.0

        if use_amp:
            scaler.scale(loss_selfplay).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_selfplay.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss = loss_expert.item() + loss_selfplay.item()

        total_loss += loss
        total_pol_e += pol_loss_e.item()
        total_rank += rank_loss.item()
        total_anchor += anchor_loss.item()
        total_pol_s += pol_loss_s.item()
        total_val_s += val_loss_s.item()
        n += 1

        # Memory diagnostics: first 3 steps + every log_interval
        if bi < 3 or (bi + 1) % log_interval == 0:
            alloc = torch.cuda.memory_allocated() / 1e9
            resv = torch.cuda.memory_reserved() / 1e9
            if bi < 3:
                print(f"  [step {bi}] GPU: {alloc:.1f} GB alloc, "
                      f"{resv:.1f} GB reserved", flush=True)

        if (bi + 1) % log_interval == 0:
            elapsed = time.time() - t0
            sps = (bi + 1) * expert_loader.batch_size * 2 / elapsed
            eta = (len(expert_loader) - bi - 1) / max(bi + 1, 1) * elapsed
            print(f"  [{bi+1}/{len(expert_loader)}] "
                  f"loss={total_loss/n:.4f} "
                  f"(e_pol={total_pol_e/n:.4f} rank={total_rank/n:.4f} "
                  f"anchor={total_anchor/n:.1f} | "
                  f"s_pol={total_pol_s/n:.4f} s_val={total_val_s/n:.1f}) "
                  f"{sps:.0f} s/s ETA {eta/60:.0f}m", flush=True)

    return (total_loss / n, total_pol_e / n, total_rank / n,
            total_anchor / n, total_pol_s / n, total_val_s / n)


@torch.no_grad()
def validate_hybrid(model, expert_val_loader, selfplay_val_loader, device,
                    max_score=500.0, num_value_bins=64, use_amp=False):
    """Validate on both expert and self-play val sets."""
    model.eval()
    anchor_bins = torch.linspace(0, max_score, num_value_bins, device=device)

    # Expert validation: policy CE + anchor MAE
    ep, em, en = 0, 0, 0
    for obs, pol, val in expert_val_loader:
        obs, pol, val = obs.to(device), pol.to(device), val.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            pol_logits, val_logits = model(obs)
            pol_loss = cross_entropy_soft(pol_logits, pol)
        pred = model.predict_value(val_logits, max_val=max_score)
        true = (val * anchor_bins).sum(dim=-1)
        mae = (pred - true).abs().mean()
        ep += pol_loss.item()
        em += mae.item()
        en += 1

    # Self-play validation: policy CE + value MSE + MAE
    sp, sv, sm, sn = 0, 0, 0, 0
    for obs, pol, val in selfplay_val_loader:
        obs, pol, val = obs.to(device), pol.to(device), val.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            pol_logits, val_logits = model(obs)
            pol_loss = cross_entropy_soft(pol_logits, pol)
            pred = model.predict_value(val_logits, max_val=max_score)
            val_loss = F.mse_loss(pred, val)
        mae = (pred - val).abs().mean()
        sp += pol_loss.item()
        sv += val_loss.item()
        sm += mae.item()
        sn += 1

    return {
        'expert_pol': ep / max(en, 1),
        'expert_mae': em / max(en, 1),
        'selfplay_pol': sp / max(sn, 1),
        'selfplay_val': sv / max(sn, 1),
        'selfplay_mae': sm / max(sn, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--expert-file', required=True,
                   help='Expert pairwise tensor (alphatrain_pairwise.pt)')
    p.add_argument('--selfplay-file', required=True,
                   help='Self-play tensor (selfplay_iter2.pt)')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--val-weight', type=float, default=0.001)
    p.add_argument('--rank-weight', type=float, default=1.0)
    p.add_argument('--anchor-weight', type=float, default=0.001)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--value-bins', type=int, default=1,
                   help='Value head size (1=scalar)')
    p.add_argument('--val-split', type=float, default=0.05)
    p.add_argument('--gpu-data', action='store_true')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--warm-start', action='store_true')
    p.add_argument('--save-dir', default='checkpoints/pillar2g')
    p.add_argument('--copy-to', type=str, default=None)
    args = p.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu')
    print(f"Device: {device}", flush=True)

    if not (args.gpu_data and device.type in ('cuda', 'mps')):
        raise NotImplementedError("Use --gpu-data with CUDA or MPS")

    # ── Load expert dataset (pairwise) ──
    print("\n--- Expert dataset ---", flush=True)
    expert_ds = TensorDatasetGPU(args.expert_file, augment=True,
                                  device=str(device))
    assert expert_ds.has_pairs, "Expert file must have pairwise data"

    n_eval_expert = int(len(expert_ds) * args.val_split)
    n_train_expert = len(expert_ds) - n_eval_expert
    expert_train, expert_val = random_split(
        expert_ds, [n_train_expert, n_eval_expert],
        generator=torch.Generator().manual_seed(42))

    expert_train_loader = DataLoader(
        expert_train, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=expert_ds.collate_pairwise)
    expert_val_loader = DataLoader(
        expert_val, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=expert_ds.collate)

    # ── Load self-play dataset ──
    # CPU-resident: the dense policy tensor (346K × 6561 × f32 = 9 GB) doesn't
    # fit on GPU alongside the expert dataset. Collate runs on CPU (array indexing
    # + dihedral permutation), then .to(device) transfers ~13 MB per batch.
    print("\n--- Self-play dataset (CPU-resident, 9 GB policy too large for GPU) ---",
          flush=True)
    selfplay_ds = SelfPlayDataset(args.selfplay_file, augment=True,
                                   device='cpu')

    n_val_sp = int(len(selfplay_ds) * args.val_split)
    n_train_sp = len(selfplay_ds) - n_val_sp
    selfplay_train, selfplay_val = random_split(
        selfplay_ds, [n_train_sp, n_val_sp],
        generator=torch.Generator().manual_seed(42))

    selfplay_train_loader = DataLoader(
        selfplay_train, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=selfplay_ds.collate)
    selfplay_val_loader = DataLoader(
        selfplay_val, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=selfplay_ds.collate)

    max_score = expert_ds.max_score

    if device.type == 'cuda':
        alloc = torch.cuda.memory_allocated() / 1e9
        resv = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU after data load: {alloc:.1f} GB alloc, {resv:.1f} GB reserved",
              flush=True)
    print(f"\nExpert: {n_train_expert:,} train, {n_eval_expert:,} val "
          f"(pairwise={expert_ds.has_pairs})", flush=True)
    print(f"Self-play: {n_train_sp:,} train, {n_val_sp:,} val", flush=True)
    print(f"Expert loader: {len(expert_train_loader)} batches/epoch", flush=True)
    print(f"Self-play loader: {len(selfplay_train_loader)} batches/epoch "
          f"(cycled to {len(expert_train_loader)})", flush=True)

    # ── Model ──
    model = AlphaTrainNet(num_blocks=args.num_blocks, channels=args.channels,
                          num_value_bins=args.value_bins).to(device)
    n_params = count_parameters(model)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    print(f"Model: {args.num_blocks}b x {args.channels}ch, "
          f"{n_params:,} params", flush=True)

    # ── Load weights ──
    start_epoch = 0
    best_val = float('inf')

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt['model']
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model_state = model.state_dict()
        filtered, skipped = {}, []
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        missing, _ = model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}", flush=True)
        if missing:
            print(f"  Randomly init: "
                  f"{[k for k in missing if k not in skipped]}", flush=True)
        if not args.warm_start and 'optimizer' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            best_val = ckpt.get('best_val_loss', float('inf'))
        print(f"Warm start from epoch {ckpt.get('epoch', '?')}", flush=True)

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("torch.compile enabled", flush=True)

    train_params = [pp for pp in model.parameters() if pp.requires_grad]
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

    os.makedirs(args.save_dir, exist_ok=True)
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("AMP enabled", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Hybrid training: {args.epochs} epochs", flush=True)
    print(f"  Expert: pol_CE + {args.val_weight} * "
          f"({args.rank_weight} * rank + {args.anchor_weight} * anchor)",
          flush=True)
    print(f"  Self-play: pol_CE + {args.val_weight} * MSE", flush=True)
    print(f"  LR: {args.lr}, warmup: {args.warmup_epochs} epochs", flush=True)
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        et = time.time()
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.2e})", flush=True)

        tl, ep_l, rk, an, sp_l, sv_l = train_epoch_hybrid(
            model, expert_train_loader, selfplay_train_loader,
            optimizer, device, max_score=max_score,
            num_value_bins=expert_ds.num_value_bins,
            scaler=scaler, val_weight=args.val_weight,
            rank_weight=args.rank_weight, anchor_weight=args.anchor_weight)

        val = validate_hybrid(
            model, expert_val_loader, selfplay_val_loader,
            device, max_score=max_score,
            num_value_bins=expert_ds.num_value_bins, use_amp=use_amp)

        scheduler.step()

        print(f"  Train: loss={tl:.4f} "
              f"(e_pol={ep_l:.4f} rank={rk:.4f} anchor={an:.1f} | "
              f"s_pol={sp_l:.4f} s_val={sv_l:.1f})", flush=True)
        print(f"  Val:   e_pol={val['expert_pol']:.4f} "
              f"e_mae={val['expert_mae']:.0f} | "
              f"s_pol={val['selfplay_pol']:.4f} "
              f"s_val={val['selfplay_val']:.1f} "
              f"s_mae={val['selfplay_mae']:.0f} "
              f"[{time.time()-et:.0f}s]", flush=True)

        # Use combined metric for best model selection:
        # expert policy + selfplay policy (both should be good)
        val_combined = val['expert_pol'] + val['selfplay_pol']

        ckpt_data = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_metrics': val,
            'val_loss': val_combined,
            'best_val_loss': min(best_val, val_combined),
            'max_score': max_score,
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
                         os.path.join(copy_dir, 'alphatrain_td_latest.pt'))
            shutil.copy2(epoch_path,
                         os.path.join(copy_dir,
                                      f'alphatrain_td_epoch_{epoch+1}.pt'))

        if val_combined < best_val:
            best_val = val_combined
            best_path = os.path.join(args.save_dir, 'best.pt')
            torch.save(ckpt_data, best_path)
            print(f"  ** New best (combined_pol={val_combined:.4f}) **",
                  flush=True)
            if args.copy_to:
                shutil.copy2(best_path, args.copy_to)
                print(f"  Copied to {args.copy_to}", flush=True)

    print(f"\n=== Done: {args.epochs} epochs in "
          f"{(time.time()-t0)/3600:.1f}h ===", flush=True)
    print(f"Best combined pol loss: {best_val:.4f}", flush=True)


if __name__ == '__main__':
    main()
