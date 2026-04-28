"""Train a ranking readout on frozen policy backbone.

Extracts afterstate features from the frozen backbone and trains a
scalar readout to rank sibling moves by MCTS visit preference.

Loss: margin-weighted pairwise — score(better) > score(worse).
Split: game-level holdout (no leakage).

Usage:
    python -m alphatrain.scripts.train_ranking_head \
        --model alphatrain/data/pillar2w2_epoch_10.pt \
        --dirs data/selfplay_v7_s1600 data/selfplay_v8_s1600 data/crisis_v2 \
        --output alphatrain/data/ranking_head.pt \
        --device mps
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation
from game.board import _clear_lines_at


def build_obs_from_board(board, next_balls):
    next_r = np.zeros(3, dtype=np.intp)
    next_c = np.zeros(3, dtype=np.intp)
    next_color = np.zeros(3, dtype=np.intp)
    n_next = min(len(next_balls), 3)
    for i in range(n_next):
        nb = next_balls[i]
        next_r[i] = nb['row']
        next_c[i] = nb['col']
        next_color[i] = nb['color']
    return build_observation(board, next_r, next_c, next_color, n_next)


def make_afterstate(board, sr, sc, tr, tc):
    """Create afterstate: move ball + clear any lines formed."""
    b = board.copy()
    b[tr, tc] = b[sr, sc]
    b[sr, sc] = 0
    _clear_lines_at(b, tr, tc)
    return b


class RankingHead(nn.Module):
    """Scalar readout on frozen backbone features (global-avg-pooled).

    hidden=0: linear (256→1)
    hidden>0: MLP (256→hidden→1)
    """

    def __init__(self, in_features=256, hidden=0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1))
        else:
            self.net = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.net(x)


class SpatialRankingHead(nn.Module):
    """Spatial ranking head matching the garbage value head architecture.

    Conv2d(channels→value_ch, 1x1) → BN → ReLU → flatten → FC → ReLU → FC → 1
    Preserves spatial structure instead of global-average-pooling it away.
    """

    def __init__(self, channels=256, value_channels=32, value_hidden=256,
                 dropout=0.3):
        super().__init__()
        self.conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(value_channels)
        self.fc1 = nn.Linear(value_channels * 9 * 9, value_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(value_hidden, 1)
        # Match the init used in randomize_value_head.py
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        """x: (batch, 256, 9, 9) backbone features → (batch, 1)"""
        v = F.relu(self.bn(self.conv(x)))
        v = v.reshape(v.size(0), -1)
        v = self.dropout(F.relu(self.fc1(v)))
        return self.fc2(v)


def extract_pairs_from_game(path):
    """Extract afterstate observation pairs with visit margins."""
    data = json.load(open(path))
    moves = data['moves']
    step = max(1, len(moves) // 100)
    pairs = []

    for turn in range(0, len(moves), step):
        move_data = moves[turn]
        top_moves = move_data.get('top_moves', [])
        top_scores = move_data.get('top_scores', [])
        if len(top_moves) < 2:
            continue

        board = np.array(move_data['board'], dtype=np.int8)
        nb = move_data['next_balls']

        m0 = top_moves[0]
        after_0 = make_afterstate(
            board, m0['sr'], m0['sc'], m0['tr'], m0['tc'])
        obs_0 = build_obs_from_board(after_0, nb)

        for i in range(1, len(top_moves)):
            mi = top_moves[i]
            margin = top_scores[0] - top_scores[i]
            after_i = make_afterstate(
                board, mi['sr'], mi['sc'], mi['tr'], mi['tc'])
            obs_i = build_obs_from_board(after_i, nb)
            pairs.append((obs_0, obs_i, margin))

    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.obs_better = torch.from_numpy(
            np.array([p[0] for p in pairs], dtype=np.float32))
        self.obs_worse = torch.from_numpy(
            np.array([p[1] for p in pairs], dtype=np.float32))
        self.margins = torch.tensor(
            [p[2] for p in pairs], dtype=torch.float32)

    def __len__(self):
        return len(self.margins)

    def __getitem__(self, idx):
        return self.obs_better[idx], self.obs_worse[idx], self.margins[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dirs', nargs='+', default=None,
                   help='Game JSON dirs (builds observations on the fly)')
    p.add_argument('--pairs-file', default=None,
                   help='Pre-computed pairs .pt file (from build_ranking_data_exact)')
    p.add_argument('--output', default='alphatrain/data/ranking_head.pt')
    p.add_argument('--max-games', type=int, default=500)
    p.add_argument('--max-pairs', type=int, default=200000)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=0,
                   help='Hidden dim for MLP (0 = linear head)')
    p.add_argument('--spatial', action='store_true',
                   help='Use spatial head (conv+FC, matches garbage head arch)')
    p.add_argument('--frozen-spatial', action='store_true',
                   help='Freeze random spatial layers, train only final FC')
    p.add_argument('--value-channels', type=int, default=8,
                   help='Channels for spatial conv (garbage head uses 32)')
    p.add_argument('--value-hidden', type=int, default=64,
                   help='Hidden dim for spatial FC (garbage head uses 512)')
    p.add_argument('--device', default='mps')
    p.add_argument('--val-fraction', type=float, default=0.2)
    args = p.parse_args()

    device = torch.device(args.device)
    net, max_score = load_model(args.model, device, fp16=False, jit_trace=False)
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    print(f"Backbone frozen on {device}", flush=True)

    if args.pairs_file:
        # Load pre-computed exact pairs
        print(f"Loading {args.pairs_file}...", flush=True)
        pdata = torch.load(args.pairs_file, weights_only=False)
        n_total = len(pdata['margins'])

        # Game-level split
        if 'game_ids' in pdata:
            gids = pdata['game_ids'].numpy()
            unique_games = np.unique(gids)
            rng = np.random.RandomState(42)
            rng.shuffle(unique_games)
            split_idx = int(len(unique_games) * (1 - args.val_fraction))
            train_games = set(unique_games[:split_idx].tolist())
            train_mask = np.array([g in train_games for g in gids])
        else:
            # Fallback: split by index
            train_mask = np.zeros(n_total, dtype=bool)
            train_mask[:int(n_total * (1 - args.val_fraction))] = True

        val_mask = ~train_mask
        train_set = PairDataset(list(zip(
            pdata['obs_better'][train_mask].numpy(),
            pdata['obs_worse'][train_mask].numpy(),
            pdata['margins'][train_mask].tolist())))
        val_set = PairDataset(list(zip(
            pdata['obs_better'][val_mask].numpy(),
            pdata['obs_worse'][val_mask].numpy(),
            pdata['margins'][val_mask].tolist())))
        print(f"Pairs: {len(train_set)} train, {len(val_set)} val "
              f"(game-level split)", flush=True)
    else:
        assert args.dirs, "Must provide --dirs or --pairs-file"
        # Collect and split by game
        all_files = []
        for d in args.dirs:
            files = sorted(f for f in os.listdir(d) if f.endswith('.json'))
            all_files.extend(os.path.join(d, f) for f in files)
        all_files = all_files[:args.max_games]
        rng = np.random.RandomState(42)
        rng.shuffle(all_files)
        split = int(len(all_files) * (1 - args.val_fraction))
        train_files = all_files[:split]
        val_files = all_files[split:]
        print(f"Games: {len(train_files)} train, {len(val_files)} val",
              flush=True)

        t0 = time.time()
        train_pairs = []
        for i, path in enumerate(train_files):
            train_pairs.extend(extract_pairs_from_game(path))
            if len(train_pairs) >= args.max_pairs:
                train_pairs = train_pairs[:args.max_pairs]
                break
            if (i + 1) % 100 == 0:
                print(f"  train [{i+1}/{len(train_files)}] "
                      f"{len(train_pairs)} pairs", flush=True)

        val_pairs = []
        for path in val_files[:100]:
            val_pairs.extend(extract_pairs_from_game(path))

        print(f"Pairs: {len(train_pairs)} train, {len(val_pairs)} val "
              f"({time.time()-t0:.0f}s)", flush=True)
        train_set = PairDataset(train_pairs)
        val_set = PairDataset(val_pairs)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Ranking head
    use_spatial = args.spatial
    if use_spatial:
        head = SpatialRankingHead(channels=256,
                                  value_channels=args.value_channels,
                                  value_hidden=args.value_hidden).to(device)
        if args.frozen_spatial:
            # Freeze conv + bn + fc1 (random spatial mixer)
            # Only fc2 is trainable
            for name, param in head.named_parameters():
                if 'fc2' not in name:
                    param.requires_grad_(False)
    else:
        head = RankingHead(in_features=256, hidden=args.hidden).to(device)
    n_trainable = sum(pp.numel() for pp in head.parameters() if pp.requires_grad)
    n_total = sum(pp.numel() for pp in head.parameters())
    print(f"{'Frozen spatial' if args.frozen_spatial else 'Spatial' if use_spatial else 'Linear'} "
          f"head: {n_trainable:,} trainable / {n_total:,} total params",
          flush=True)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    def extract_features(obs_batch):
        with torch.no_grad():
            out = net.stem(obs_batch)
            out = net.blocks(out)
            out = F.relu(net.backbone_bn(out))
        out = out.detach()
        if use_spatial:
            return out  # (batch, 256, 9, 9)
        return out.mean(dim=(2, 3))  # (batch, 256)

    best_val_acc = 0.0

    frozen_spatial = use_spatial and args.frozen_spatial

    for epoch in range(args.epochs):
        if frozen_spatial:
            # Only fc2 trains — keep everything else in eval mode
            head.eval()
            head.fc2.train()
        else:
            head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t_epoch = time.time()

        for obs_better, obs_worse, margin in train_loader:
            obs_better = obs_better.to(device)
            obs_worse = obs_worse.to(device)
            margin = margin.to(device)

            feat_better = extract_features(obs_better)
            feat_worse = extract_features(obs_worse)

            score_better = head(feat_better).squeeze(-1)
            score_worse = head(feat_worse).squeeze(-1)

            diff = score_better - score_worse
            weight = 1.0 + margin.clamp(0, 3)
            loss = torch.log1p(torch.exp(-diff * weight)).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(margin)
            train_correct += (diff > 0).sum().item()
            train_total += len(margin)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for obs_better, obs_worse, margin in val_loader:
                obs_better = obs_better.to(device)
                obs_worse = obs_worse.to(device)
                margin = margin.to(device)

                feat_better = extract_features(obs_better)
                feat_worse = extract_features(obs_worse)

                score_better = head(feat_better).squeeze(-1)
                score_worse = head(feat_worse).squeeze(-1)

                diff = score_better - score_worse
                weight = 1.0 + margin.clamp(0, 3)
                loss = torch.log1p(torch.exp(-diff * weight)).mean()

                val_loss += loss.item() * len(margin)
                val_correct += (diff > 0).sum().item()
                val_total += len(margin)

        val_loss /= val_total
        val_acc = val_correct / val_total

        elapsed = time.time() - t_epoch
        improved = val_acc > best_val_acc
        tag = " *" if improved else ""
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"loss={train_loss:.4f} acc={100*train_acc:.1f}% | "
              f"val_loss={val_loss:.4f} val_acc={100*val_acc:.1f}% "
              f"({elapsed:.0f}s){tag}", flush=True)

        ckpt = {
            'head': head.state_dict(),
            'hidden': args.hidden,
            'spatial': use_spatial,
            'value_channels': args.value_channels if use_spatial else 0,
            'value_hidden': args.value_hidden if use_spatial else 0,
            'in_features': 256,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch': epoch + 1,
        }

        base, ext = os.path.splitext(args.output)
        torch.save(ckpt, f"{base}_epoch_{epoch+1}{ext}")

        if improved:
            best_val_acc = val_acc
            torch.save(ckpt, args.output)

    print(f"\nDone. Best val_acc={100*best_val_acc:.1f}%", flush=True)


if __name__ == '__main__':
    main()
