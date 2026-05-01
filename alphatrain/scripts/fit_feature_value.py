"""Fit linear regression on board features to predict log(1+remaining_turns).

Saves the fitted weights, feature standardization stats, and metadata to
alphatrain/data/feature_value_weights.npz for use as an MCTS leaf evaluator.

Usage:
    python -m alphatrain.scripts.fit_feature_value \
        --dirs data/selfplay_v8_combined data/crisis_v2 \
        --max-games 5000 --positions-per-game 10 \
        --out alphatrain/data/feature_value_weights.npz
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
    p.add_argument('--out', required=True)
    p.add_argument('--ridge', type=float, default=1e-3)
    args = p.parse_args()

    rng = random.Random(args.seed)

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

    feature_names_full = list(FEATURE_NAMES) + ['ratio', 'frag_score']

    X = []
    y = []

    for gi, path in enumerate(all_files):
        with open(path) as f:
            data = json.load(f)
        moves = data['moves']
        n = len(moves)
        if n < 30:
            continue
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
            empty_v = feat[FEATURE_NAMES.index('empty')]
            largest_v = feat[FEATURE_NAMES.index('largest')]
            n_comp_v = feat[FEATURE_NAMES.index('components')]
            ratio = largest_v / max(empty_v, 1)
            frag_score = (empty_v - largest_v) * n_comp_v
            feat.extend([ratio, frag_score])
            X.append(feat)
            remaining = n - idx
            y.append(math.log1p(remaining))

        if (gi + 1) % 1000 == 0:
            print(f"  [{gi+1}/{len(all_files)}] N={len(y)}", flush=True)

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    print(f"\nDataset: {X.shape[0]} positions, {X.shape[1]} features",
          flush=True)

    # Standardize
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    Xn = (X - means) / stds

    # Train/val split for reporting (NOT used to refit — final fit uses all)
    n_total = X.shape[0]
    perm = np.random.RandomState(0).permutation(n_total)
    n_train = int(0.8 * n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    Xt = Xn[train_idx]
    Xv = Xn[val_idx]
    yt = y[train_idx]
    yv = y[val_idx]

    Xt_b = np.hstack([Xt, np.ones((Xt.shape[0], 1))])
    Xv_b = np.hstack([Xv, np.ones((Xv.shape[0], 1))])
    A = Xt_b.T @ Xt_b + args.ridge * np.eye(Xt_b.shape[1])
    b = Xt_b.T @ yt
    w_train = np.linalg.solve(A, b)

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1 - ss_res / ss_tot

    yt_pred = Xt_b @ w_train
    yv_pred = Xv_b @ w_train
    print(f"Train R^2: {r2(yt, yt_pred):.4f} | "
          f"Val R^2: {r2(yv, yv_pred):.4f}", flush=True)

    # Final model: refit on ALL data so we ship the best estimate
    X_all_b = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
    A_all = X_all_b.T @ X_all_b + args.ridge * np.eye(X_all_b.shape[1])
    b_all = X_all_b.T @ y
    w_all = np.linalg.solve(A_all, b_all)
    coefs = w_all[:-1]
    bias = float(w_all[-1])

    print(f"\nFinal model bias: {bias:+.4f}", flush=True)
    print(f"Standardized coefficients (sorted by |coef|):", flush=True)
    ordered = sorted(
        [(name, coefs[i]) for i, name in enumerate(feature_names_full)],
        key=lambda x: -abs(x[1]))
    for name, c in ordered:
        bar = ('+' if c > 0 else '-') * min(int(abs(c) * 20), 40)
        print(f"  {name:<18} {c:>+8.4f}  {bar}", flush=True)

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        coefs=coefs.astype(np.float32),
        bias=np.float32(bias),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
        feature_names=np.array(feature_names_full),
        n_positions=np.int32(X.shape[0]),
        n_games=np.int32(len(all_files)),
        ridge=np.float32(args.ridge),
    )
    print(f"\nSaved to {args.out}", flush=True)


if __name__ == '__main__':
    main()
