"""Linear probe: can backbone features predict MCTS move preference?

Loads game JSONs, finds positions where policy disagrees with MCTS,
extracts backbone features for both moves' afterstates, and fits a
linear model to predict which move MCTS preferred.

This is a zero-cost gate check before training a ranking head.

Usage:
    python -m alphatrain.scripts.linear_probe_value \
        --model alphatrain/data/pillar2w2_epoch_10.pt \
        --dirs data/selfplay_v7_s1600 data/selfplay_v8_s1600 \
        --max-games 200 --device mps
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation
from alphatrain.mcts import _get_legal_priors_flat


def extract_backbone_features(net, obs_tensor):
    """Run backbone only, return (batch, 256, 9, 9) features."""
    with torch.inference_mode():
        out = net.stem(obs_tensor)
        out = net.blocks(out)
        out = F.relu(net.backbone_bn(out))
    return out


def make_afterstate(board, sr, sc, tr, tc):
    """Create afterstate: move ball + clear any lines formed."""
    from game.board import _clear_lines_at
    b = board.copy()
    b[tr, tc] = b[sr, sc]
    b[sr, sc] = 0
    _clear_lines_at(b, tr, tc)
    return b


def build_obs_from_board(board, next_balls):
    """Build observation from board array and next_balls list."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dirs', nargs='+', required=True)
    p.add_argument('--max-games', type=int, default=200)
    p.add_argument('--max-positions', type=int, default=15000,
                   help='Stop after this many positions (not pairs)')
    p.add_argument('--device', default='mps')
    args = p.parse_args()

    device = torch.device(args.device)
    net, max_score = load_model(args.model, device, fp16=False, jit_trace=False)
    net.eval()
    print(f"Model loaded on {device}", flush=True)

    # Collect game files
    all_files = []
    for d in args.dirs:
        files = sorted(f for f in os.listdir(d) if f.endswith('.json'))
        all_files.extend(os.path.join(d, f) for f in files)
    all_files = all_files[:args.max_games]
    print(f"Processing {len(all_files)} games", flush=True)

    # Extract pairs: for each position, get afterstate features for
    # MCTS top-1 move and a lower-ranked move
    feat_preferred = []   # backbone features for MCTS-preferred afterstate
    feat_other = []       # backbone features for lower-ranked afterstate
    visit_margins = []    # how much MCTS preferred move A over move B
    game_ids = []         # which game each pair came from (for game-level split)
    is_disagree = []      # whether policy disagreed with MCTS on this position
    n_agree = 0
    n_disagree = 0
    n_pairs = 0
    t0 = time.time()

    total_positions = 0
    for gi, path in enumerate(all_files):
        if total_positions >= args.max_positions:
            break
        data = json.load(open(path))
        moves = data['moves']
        # Sample every Nth turn to spread across game length
        step = max(1, len(moves) // 50)

        for turn in range(0, len(moves), step):
            if total_positions >= args.max_positions:
                break
            move_data = moves[turn]
            total_positions += 1
            board = np.array(move_data['board'], dtype=np.int8)
            nb = move_data['next_balls']
            top_moves = move_data.get('top_moves', [])
            top_scores = move_data.get('top_scores', [])

            if len(top_moves) < 2:
                continue

            # MCTS top move (highest visits)
            m0 = top_moves[0]

            # Policy top move
            obs = build_obs_from_board(board, nb)
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.inference_mode():
                pol_logits, _ = net(obs_t)
            pol_np = pol_logits[0].float().cpu().numpy()
            priors = _get_legal_priors_flat(board, pol_np, 30)
            if not priors:
                continue
            policy_action = max(priors.items(), key=lambda x: x[1])[0]
            psr = policy_action // 81 // 9
            psc = policy_action // 81 % 9
            ptr = policy_action % 81 // 9
            ptc = policy_action % 81 % 9

            pos_disagree = (m0['sr'], m0['sc'], m0['tr'], m0['tc']) != \
                           (psr, psc, ptr, ptc)
            if pos_disagree:
                n_disagree += 1
            else:
                n_agree += 1

            # Create pairs: MCTS top-1 vs each lower-ranked move
            # Use top_scores (log visit fractions) for margin
            for i in range(1, len(top_moves)):
                mi = top_moves[i]
                margin = top_scores[0] - top_scores[i]  # positive = top-1 better

                # Afterstates
                after_0 = make_afterstate(
                    board, m0['sr'], m0['sc'], m0['tr'], m0['tc'])
                after_i = make_afterstate(
                    board, mi['sr'], mi['sc'], mi['tr'], mi['tc'])

                obs_0 = build_obs_from_board(after_0, nb)
                obs_i = build_obs_from_board(after_i, nb)

                batch = torch.from_numpy(
                    np.stack([obs_0, obs_i])).to(device)
                feats = extract_backbone_features(net, batch)
                feats_pooled = feats.mean(dim=(2, 3)).cpu().numpy()

                feat_preferred.append(feats_pooled[0])
                feat_other.append(feats_pooled[1])
                visit_margins.append(margin)
                game_ids.append(gi)
                is_disagree.append(pos_disagree)
                n_pairs += 1


        if (gi + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{gi+1}/{len(all_files)}] pairs={n_pairs} "
                  f"agree={n_agree} disagree={n_disagree} ({elapsed:.0f}s)",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nExtracted {n_pairs} pairs from "
          f"{n_agree + n_disagree} positions ({elapsed:.0f}s)", flush=True)
    print(f"Disagreement rate: "
          f"{100*n_disagree/(n_agree+n_disagree+1e-9):.1f}%", flush=True)

    if n_pairs < 100:
        print("Too few pairs for a meaningful probe.", flush=True)
        return

    # Fit linear probe with GAME-LEVEL split (no leakage)
    X_pref = np.array(feat_preferred)
    X_other = np.array(feat_other)
    margins = np.array(visit_margins)
    gids = np.array(game_ids)
    disagree_flags = np.array(is_disagree)

    X_diff = X_pref - X_other  # (N, 256)

    # Game-level split: 80% of games for train, 20% for val
    unique_games = np.unique(gids)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_games)
    split_idx = int(0.8 * len(unique_games))
    train_games = set(unique_games[:split_idx])
    val_games = set(unique_games[split_idx:])

    train_mask = np.array([g in train_games for g in gids])
    val_mask = ~train_mask

    X_train = X_diff[train_mask]
    X_val = X_diff[val_mask]
    # Labels: predict sign of difference (always positive by construction)
    # Balance by also fitting on reversed pairs
    X_train_bal = np.concatenate([X_train, -X_train])
    y_train_bal = np.concatenate([np.ones(len(X_train)),
                                  np.zeros(len(X_train))])
    X_val_bal = np.concatenate([X_val, -X_val])
    y_val_bal = np.concatenate([np.ones(len(X_val)),
                                np.zeros(len(X_val))])

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_bal, y_train_bal)

    train_acc = clf.score(X_train_bal, y_train_bal)
    val_acc = clf.score(X_val_bal, y_val_bal)

    print(f"\n{'='*50}", flush=True)
    print(f"LINEAR PROBE RESULTS (game-level holdout)", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"Total pairs: {n_pairs} (from {n_agree+n_disagree} positions)",
          flush=True)
    print(f"Train games: {len(train_games)}, Val games: {len(val_games)}",
          flush=True)
    print(f"Train pairs: {train_mask.sum()}, Val pairs: {val_mask.sum()}",
          flush=True)
    print(f"Feature dim: {X_diff.shape[1]}", flush=True)
    print(f"Train accuracy: {100*train_acc:.1f}%", flush=True)
    print(f"Val accuracy:   {100*val_acc:.1f}%", flush=True)
    print(f"  (50% = random, >55% = backbone has signal)", flush=True)

    # Accuracy by margin bucket
    val_margins = margins[val_mask]
    print(f"\nVal accuracy by visit margin:", flush=True)
    for lo, hi, label in [(0, 0.5, "close"), (0.5, 1.5, "medium"),
                          (1.5, 100, "large")]:
        mask = (val_margins >= lo) & (val_margins < hi)
        if mask.sum() > 10:
            X_sub = X_val[mask]
            X_sub_bal = np.concatenate([X_sub, -X_sub])
            y_sub_bal = np.concatenate([np.ones(len(X_sub)),
                                        np.zeros(len(X_sub))])
            subset_acc = clf.score(X_sub_bal, y_sub_bal)
            print(f"  {label} ({lo}-{hi}): "
                  f"acc={100*subset_acc:.1f}% (n={mask.sum()})", flush=True)

    # Disagreement-only accuracy
    val_disagree = disagree_flags[val_mask]
    if val_disagree.sum() > 10:
        X_dis = X_val[val_disagree]
        X_dis_bal = np.concatenate([X_dis, -X_dis])
        y_dis_bal = np.concatenate([np.ones(len(X_dis)),
                                    np.zeros(len(X_dis))])
        dis_acc = clf.score(X_dis_bal, y_dis_bal)
        print(f"\n  Disagreement-only: acc={100*dis_acc:.1f}% "
              f"(n={val_disagree.sum()})", flush=True)

    # PCA analysis
    print(f"\nPCA analysis:", flush=True)
    from sklearn.decomposition import PCA
    for n_comp in [4, 8, 16, 32]:
        pca = PCA(n_components=n_comp)
        X_pca_train = pca.fit_transform(X_train_bal)
        X_pca_val = pca.transform(X_val_bal)
        clf_pca = LogisticRegression(max_iter=1000, C=1.0)
        clf_pca.fit(X_pca_train, y_train_bal)
        acc = clf_pca.score(X_pca_val, y_val_bal)
        print(f"  PCA-{n_comp}: val acc={100*acc:.1f}%", flush=True)


if __name__ == '__main__':
    main()
