"""Train a separate value network on survival targets.

Input: board positions with survival fraction targets [0, 1].
Output: ValueNet checkpoint that predicts board health.

Usage:
    python -m alphatrain.scripts.train_value \
        --data alphatrain/data/value_train.pt \
        --output alphatrain/data/value_net.pt \
        --epochs 10 --batch-size 4096 --lr 1e-3
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from alphatrain.model import ValueNet
from alphatrain.observation import build_observation


class ValueDataset(Dataset):
    """Builds observations on-the-fly from compact board storage."""

    def __init__(self, boards, next_balls, targets):
        self.boards = boards       # (N, 9, 9) int8
        self.next_balls = next_balls  # (N, 3, 3) int8 — [ball_idx, (r,c,color)]
        self.targets = targets     # (N,) float32

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        board = self.boards[idx].numpy()
        nb = self.next_balls[idx].numpy()

        # Unpack next_balls
        next_r = np.zeros(3, dtype=np.intp)
        next_c = np.zeros(3, dtype=np.intp)
        next_color = np.zeros(3, dtype=np.intp)
        n_next = 0
        for i in range(3):
            if nb[i, 2] > 0:  # has a color
                next_r[i] = nb[i, 0]
                next_c[i] = nb[i, 1]
                next_color[i] = nb[i, 2]
                n_next += 1

        obs = build_observation(board, next_r, next_c, next_color, n_next)
        return torch.from_numpy(obs), self.targets[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='alphatrain/data/value_train.pt')
    p.add_argument('--output', default='alphatrain/data/value_net.pt')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num-blocks', type=int, default=5)
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--device', default=None)
    p.add_argument('--val-fraction', type=float, default=0.05)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Load data
    print(f"Loading {args.data}...", flush=True)
    data = torch.load(args.data, weights_only=False)
    boards = data['boards']
    next_balls = data['next_balls']
    targets = data['targets']
    horizon = data['horizon']
    print(f"  {len(targets):,} positions, horizon={horizon}, "
          f"target mean={targets.mean():.3f}", flush=True)

    # Split train/val
    dataset = ValueDataset(boards, next_balls, targets)
    n_val = int(len(dataset) * args.val_fraction)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))
    # Use more workers for observation building (CPU-bound)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4,
                              persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2,
                            persistent_workers=True, pin_memory=True)
    print(f"  Train: {n_train:,}, Val: {n_val:,}", flush=True)

    # Model
    net = ValueNet(in_channels=18, num_blocks=args.num_blocks,
                   channels=args.channels, num_value_bins=1)
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"ValueNet: {args.num_blocks}b x {args.channels}ch, "
          f"{n_params:,} params, device={device}", flush=True)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        net.train()
        train_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for obs, target in train_loader:
            obs = obs.to(device)
            target = target.to(device)

            logits = net(obs)
            pred = torch.sigmoid(logits.squeeze(-1))
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                print(f"  [{n_batches}] loss={train_loss/n_batches:.4f}",
                      flush=True)

        scheduler.step()
        train_loss /= n_batches

        # Validate
        net.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for obs, target in val_loader:
                obs = obs.to(device)
                target = target.to(device)

                logits = net(obs)
                pred = torch.sigmoid(logits.squeeze(-1))
                loss = F.mse_loss(pred, target)

                val_loss += loss.item()
                val_batches += 1
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

        val_loss /= val_batches
        preds = torch.cat(all_preds)
        tgts = torch.cat(all_targets)

        # Accuracy: how often does pred correctly identify dying vs healthy?
        pred_dying = (preds < 0.5).float()
        true_dying = (tgts < 0.5).float()
        accuracy = ((pred_dying == true_dying).float().mean() * 100).item()

        elapsed = time.time() - t0
        improved = val_loss < best_val_loss
        tag = " *" if improved else ""
        print(f"Epoch {epoch+1}/{args.epochs}: train={train_loss:.4f} "
              f"val={val_loss:.4f} acc={accuracy:.1f}% "
              f"lr={scheduler.get_last_lr()[0]:.1e} "
              f"({elapsed:.0f}s){tag}", flush=True)

        ckpt = {
            'model': net.state_dict(),
            'num_blocks': args.num_blocks,
            'channels': args.channels,
            'horizon': horizon,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'epoch': epoch + 1,
        }

        # Per-epoch save
        base, ext = os.path.splitext(args.output)
        epoch_path = f"{base}_epoch_{epoch+1}{ext}"
        torch.save(ckpt, epoch_path)

        if improved:
            best_val_loss = val_loss
            torch.save(ckpt, args.output)
            print(f"  Saved: {args.output} + {epoch_path}", flush=True)
        else:
            print(f"  Saved: {epoch_path}", flush=True)

    print(f"\nDone. Best val_loss={best_val_loss:.4f}", flush=True)


if __name__ == '__main__':
    main()
