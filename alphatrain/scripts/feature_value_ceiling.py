"""Find the upper bound of board features as a value predictor.

Samples board states from existing self-play data, extracts the 16
features from mine_death_features, and reports:
  - Pearson correlation of each feature with log(1 + remaining_turns)
  - R^2 of a linear regression on all features (the linear ceiling)
  - Top single-feature predictor

If R^2 is high enough (>0.5), we can ship a linear value evaluator
without mining more features.

Usage:
    python -m alphatrain.scripts.feature_value_ceiling \
        --dirs data/selfplay_v8_combined data/crisis_v2 \
        --max-games 5000 --positions-per-game 10
"""

import os
import json
import argparse
import math
import random

import numpy as np

from alphatrain.scripts.mine_death_features import (
    board_features, FEATURE_NAMES,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', nargs='+', required=True)
    p.add_argument('--max-games', type=int, default=5000)
    p.add_argument('--positions-per-game', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)

    # Collect file list
    all_files = []
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"  WARN: missing dir {d}", flush=True)
            continue
        files = [os.path.join(d, f) for f in os.listdir(d)
                 if f.endswith('.json')]
        all_files.extend(files)
    rng.shuffle(all_files)
    all_files = all_files[:args.max_games]
    print(f"Sampling from {len(all_files)} games", flush=True)

    # Derived feature names (added in board_features wrapper below)
    derived = ['ratio', 'frag_score', 'log_remaining']
    feature_names_full = list(FEATURE_NAMES) + derived[:-1]  # excl. label

    X = []  # features
    y = []  # log(1 + remaining_turns)
    capped_count = 0
    short_skipped = 0

    for gi, path in enumerate(all_files):
        with open(path) as f:
            data = json.load(f)
        moves = data['moves']
        n = len(moves)
        if n < 30:
            short_skipped += 1
            continue
        if data.get('capped', False):
            capped_count += 1

        # Sample positions uniformly across the game
        # Avoid the very last few moves (label collapses)
        max_idx = n - 5
        if max_idx <= 5:
            continue
        sample_idxs = rng.sample(
            range(5, max_idx),
            min(args.positions_per_game, max_idx - 5))

        for idx in sample_idxs:
            board = np.array(moves[idx]['board'], dtype=np.int8)
            raw = board_features(board)
            feat = list(raw)
            # Derived features
            empty_v = feat[FEATURE_NAMES.index('empty')]
            largest_v = feat[FEATURE_NAMES.index('largest')]
            n_comp_v = feat[FEATURE_NAMES.index('components')]
            ratio = largest_v / max(empty_v, 1)
            frag_score = (empty_v - largest_v) * n_comp_v
            feat.extend([ratio, frag_score])

            remaining = n - idx
            label = math.log1p(remaining)

            X.append(feat)
            y.append(label)

        if (gi + 1) % 1000 == 0:
            print(f"  [{gi+1}/{len(all_files)}] X={len(X)}", flush=True)

    if not X:
        print("No data collected!", flush=True)
        return

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    print(f"\nDataset: {X.shape[0]} positions, {X.shape[1]} features", flush=True)
    print(f"Capped games: {capped_count} ({100*capped_count/len(all_files):.0f}%) | "
          f"Short skipped: {short_skipped}", flush=True)
    print(f"Label range: log(1+rem)  min={y.min():.2f} max={y.max():.2f} "
          f"mean={y.mean():.2f}", flush=True)

    # Standardize features (robust)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    Xn = (X - means) / stds

    # ===== Per-feature Pearson correlation =====
    print(f"\n{'='*70}", flush=True)
    print(f"Per-feature Pearson r with log(1+remaining_turns)", flush=True)
    print(f"{'='*70}", flush=True)
    correlations = []
    for i, name in enumerate(feature_names_full):
        r = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append((name, r))
    correlations.sort(key=lambda x: -abs(x[1]))
    print(f"{'feature':<18} {'pearson r':>10} {'|r|':>8}", flush=True)
    print('-' * 40, flush=True)
    for name, r in correlations:
        bar = '█' * int(abs(r) * 30)
        print(f"{name:<18} {r:>+10.3f} {abs(r):>8.3f}  {bar}", flush=True)

    # ===== Linear regression: R^2 ceiling on all features =====
    # Solve normal equations with random 80/20 split
    n_total = X.shape[0]
    perm = np.random.RandomState(0).permutation(n_total)
    n_train = int(0.8 * n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    Xt = Xn[train_idx]
    yt = y[train_idx]
    Xv = Xn[val_idx]
    yv = y[val_idx]

    # Add bias column
    Xt_b = np.hstack([Xt, np.ones((Xt.shape[0], 1))])
    Xv_b = np.hstack([Xv, np.ones((Xv.shape[0], 1))])

    # Ridge regression for stability
    lam = 1e-3
    A = Xt_b.T @ Xt_b + lam * np.eye(Xt_b.shape[1])
    b = Xt_b.T @ yt
    w = np.linalg.solve(A, b)

    yt_pred = Xt_b @ w
    yv_pred = Xv_b @ w

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1 - ss_res / ss_tot

    print(f"\n{'='*70}", flush=True)
    print(f"LINEAR REGRESSION (all {len(feature_names_full)} features)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Train R^2: {r2(yt, yt_pred):.4f}", flush=True)
    print(f"Val   R^2: {r2(yv, yv_pred):.4f}", flush=True)

    # Show coefficient magnitudes (in standardized units)
    coefs = sorted(
        [(name, w[i]) for i, name in enumerate(feature_names_full)],
        key=lambda x: -abs(x[1]))
    print(f"\nTop coefficients (standardized):", flush=True)
    for name, c in coefs[:10]:
        bar = ('█' if c > 0 else '░') * min(int(abs(c) * 20), 40)
        print(f"  {name:<18} {c:>+8.3f}  {bar}", flush=True)

    # ===== Single-feature regression =====
    print(f"\n{'='*70}", flush=True)
    print(f"SINGLE-FEATURE REGRESSION (top 5 features)", flush=True)
    print(f"{'='*70}", flush=True)
    for name, _ in correlations[:5]:
        i = feature_names_full.index(name)
        Xt_s = np.hstack([Xn[train_idx, i:i+1], np.ones((n_train, 1))])
        Xv_s = np.hstack([Xn[val_idx, i:i+1], np.ones((n_total - n_train, 1))])
        w_s = np.linalg.solve(
            Xt_s.T @ Xt_s + 1e-3 * np.eye(2), Xt_s.T @ yt)
        yv_pred_s = Xv_s @ w_s
        r2_s = r2(yv, yv_pred_s)
        print(f"  {name:<18}  val R^2 = {r2_s:.4f}", flush=True)

    # ===== Top 3 features combined =====
    top3_names = [c[0] for c in correlations[:3]]
    top3_idx = [feature_names_full.index(n) for n in top3_names]
    Xt_3 = np.hstack([Xn[train_idx][:, top3_idx], np.ones((n_train, 1))])
    Xv_3 = np.hstack([Xn[val_idx][:, top3_idx], np.ones((n_total - n_train, 1))])
    w_3 = np.linalg.solve(
        Xt_3.T @ Xt_3 + 1e-3 * np.eye(4), Xt_3.T @ yt)
    yv_pred_3 = Xv_3 @ w_3
    print(f"\nTop-3 combined ({', '.join(top3_names)}): "
          f"val R^2 = {r2(yv, yv_pred_3):.4f}", flush=True)


if __name__ == '__main__':
    main()
