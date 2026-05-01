"""Compare feature-based vs NN value-head as a position evaluator.

Same sampling as feature_value_ceiling.py, but also runs the NN value
head on each board and compares R^2 against log(1 + remaining_turns).

Tells us whether shipping a feature-based leaf eval would beat the
random NN value head currently used in MCTS.

Usage:
    python -m alphatrain.scripts.feature_vs_nn_value \
        --dirs data/selfplay_v8_combined data/crisis_v2 \
        --model alphatrain/data/pillar2w2_epoch_10.pt \
        --max-games 5000 --positions-per-game 10 --device mps
"""

import os
import json
import argparse
import math
import random

import numpy as np
import torch

from alphatrain.scripts.mine_death_features import (
    board_features, FEATURE_NAMES,
)
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model


def set_eval_mode(model):
    """Set model to inference mode without using model.eval() syntax."""
    model.train(False)
    for m in model.modules():
        m.train(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', nargs='+', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--max-games', type=int, default=5000)
    p.add_argument('--positions-per-game', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)
    net, max_score = load_model(args.model, device, fp16=False, jit_trace=False)
    set_eval_mode(net)
    print(f"Loaded model, max_score={max_score:.0f}", flush=True)

    # Collect file list
    all_files = []
    for d in args.dirs:
        if not os.path.isdir(d):
            continue
        files = [os.path.join(d, f) for f in os.listdir(d)
                 if f.endswith('.json')]
        all_files.extend(files)
    rng.shuffle(all_files)
    all_files = all_files[:args.max_games]
    print(f"Sampling from {len(all_files)} games", flush=True)

    feature_names_full = list(FEATURE_NAMES) + ['ratio', 'frag_score']

    X_feat = []
    X_obs = []
    y = []

    nr_buf = np.zeros(3, dtype=np.intp)
    nc_buf = np.zeros(3, dtype=np.intp)
    ncol_buf = np.zeros(3, dtype=np.intp)

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
            move = moves[idx]
            board = np.array(move['board'], dtype=np.int8)

            # Features
            raw = board_features(board)
            feat = list(raw)
            empty_v = feat[FEATURE_NAMES.index('empty')]
            largest_v = feat[FEATURE_NAMES.index('largest')]
            n_comp_v = feat[FEATURE_NAMES.index('components')]
            ratio = largest_v / max(empty_v, 1)
            frag_score = (empty_v - largest_v) * n_comp_v
            feat.extend([ratio, frag_score])
            X_feat.append(feat)

            # Observation for NN
            nb = move['next_balls']
            nn_count = min(len(nb), 3)
            for i in range(3):
                if i < nn_count:
                    nr_buf[i] = nb[i]['row']
                    nc_buf[i] = nb[i]['col']
                    ncol_buf[i] = nb[i]['color']
                else:
                    nr_buf[i] = 0
                    nc_buf[i] = 0
                    ncol_buf[i] = 0
            obs = build_observation(board, nr_buf, nc_buf, ncol_buf, nn_count)
            X_obs.append(obs)

            remaining = n - idx
            y.append(math.log1p(remaining))

        if (gi + 1) % 1000 == 0:
            print(f"  [{gi+1}/{len(all_files)}] N={len(y)}", flush=True)

    X_feat = np.array(X_feat, dtype=np.float64)
    X_obs = np.array(X_obs, dtype=np.float32)
    y = np.array(y, dtype=np.float64)
    print(f"\nDataset: {X_feat.shape[0]} positions", flush=True)

    # Run NN forward in batches
    print(f"Running NN forward...", flush=True)
    nn_values = np.zeros(len(y), dtype=np.float64)
    bs = args.batch_size
    with torch.inference_mode():
        for i in range(0, len(y), bs):
            batch = torch.from_numpy(X_obs[i:i+bs]).to(device)
            _, val_logits = net(batch)
            v = net.predict_value(val_logits, max_val=max_score)
            nn_values[i:i+bs] = v.float().cpu().numpy()
    print(f"NN value range: [{nn_values.min():.1f}, {nn_values.max():.1f}], "
          f"mean={nn_values.mean():.1f}", flush=True)

    n_total = X_feat.shape[0]
    perm = np.random.RandomState(0).permutation(n_total)
    n_train = int(0.8 * n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1 - ss_res / ss_tot

    yt = y[train_idx]
    yv = y[val_idx]

    nn_t = nn_values[train_idx]
    nn_v = nn_values[val_idx]

    r_nn = np.corrcoef(nn_values, y)[0, 1]
    print(f"\n{'='*70}", flush=True)
    print(f"NN VALUE HEAD as evaluator", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Pearson r (NN value vs log(1+rem)): {r_nn:+.4f}", flush=True)

    nn_t_b = np.column_stack([nn_t, np.ones_like(nn_t)])
    nn_v_b = np.column_stack([nn_v, np.ones_like(nn_v)])
    w_nn = np.linalg.solve(
        nn_t_b.T @ nn_t_b + 1e-3 * np.eye(2), nn_t_b.T @ yt)
    yv_pred_nn = nn_v_b @ w_nn
    r2_nn = r2(yv, yv_pred_nn)
    print(f"Val R^2 (linear fit on NN value): {r2_nn:.4f}", flush=True)

    means = X_feat.mean(axis=0)
    stds = X_feat.std(axis=0)
    stds[stds == 0] = 1.0
    Xn = (X_feat - means) / stds

    Xt = Xn[train_idx]
    Xv = Xn[val_idx]
    Xt_b = np.hstack([Xt, np.ones((Xt.shape[0], 1))])
    Xv_b = np.hstack([Xv, np.ones((Xv.shape[0], 1))])

    A = Xt_b.T @ Xt_b + 1e-3 * np.eye(Xt_b.shape[1])
    b = Xt_b.T @ yt
    w_feat = np.linalg.solve(A, b)
    yv_pred_feat = Xv_b @ w_feat
    r2_feat = r2(yv, yv_pred_feat)

    print(f"\n{'='*70}", flush=True)
    print(f"FEATURES (all 18) as evaluator", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Val R^2 (linear fit on 18 features): {r2_feat:.4f}", flush=True)

    Xt_combined = np.column_stack([Xt, nn_t])
    Xv_combined = np.column_stack([Xv, nn_v])
    nn_mean = nn_t.mean()
    nn_std = nn_t.std() if nn_t.std() > 0 else 1.0
    Xt_combined[:, -1] = (nn_t - nn_mean) / nn_std
    Xv_combined[:, -1] = (nn_v - nn_mean) / nn_std

    Xt_cb = np.hstack([Xt_combined, np.ones((Xt_combined.shape[0], 1))])
    Xv_cb = np.hstack([Xv_combined, np.ones((Xv_combined.shape[0], 1))])
    A2 = Xt_cb.T @ Xt_cb + 1e-3 * np.eye(Xt_cb.shape[1])
    b2 = Xt_cb.T @ yt
    w_combined = np.linalg.solve(A2, b2)
    yv_pred_combined = Xv_cb @ w_combined
    r2_combined = r2(yv, yv_pred_combined)
    nn_coef = w_combined[-2]
    print(f"\n{'='*70}", flush=True)
    print(f"FEATURES + NN VALUE combined", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Val R^2 (features + NN combined): {r2_combined:.4f}", flush=True)
    print(f"NN value coefficient (standardized): {nn_coef:+.3f}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"VERDICT", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  NN value alone:    R^2 = {r2_nn:.4f}", flush=True)
    print(f"  Features alone:    R^2 = {r2_feat:.4f}", flush=True)
    print(f"  Combined:          R^2 = {r2_combined:.4f}", flush=True)
    if r2_feat > r2_nn * 1.2:
        print(f"  -> Features beat NN value by "
              f"{r2_feat/max(r2_nn,1e-6):.2f}x. Worth shipping.", flush=True)
    elif r2_nn > r2_feat * 1.2:
        print(f"  -> NN value beats features by "
              f"{r2_nn/max(r2_feat,1e-6):.2f}x. Don't replace.", flush=True)
    else:
        boost = 100*(r2_combined-max(r2_nn,r2_feat))/max(r2_nn,r2_feat,1e-6)
        print(f"  -> Roughly tied. Combined gives +{boost:.1f}% over the "
              f"better individual. Consider hybrid.", flush=True)


if __name__ == '__main__':
    main()
