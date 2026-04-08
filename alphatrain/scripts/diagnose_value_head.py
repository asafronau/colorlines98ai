"""Forensic diagnosis of value head hallucination.

Three tests:
1. Trap Test: Does the model give high values to random chaos boards?
2. Target Analysis: Discounted points vs remaining turns — which correlates
   better with board health (empty squares)?
3. Value Distribution: What does the model predict for expert boards at
   different game stages (early, mid, endgame)?

Usage:
    python -m alphatrain.scripts.diagnose_value_head \
        --model alphatrain/data/pillar2k_best.pt \
        --tensor alphatrain/data/expert_v2_pairwise_g095.pt
"""

import argparse
import numpy as np
import torch
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--tensor', default='alphatrain/data/expert_v2_pairwise_g095.pt')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = args.device

    # Load model
    print("Loading model...", flush=True)
    from alphatrain.evaluate import load_model
    net, max_score = load_model(args.model, device)
    net.train(False)
    print(f"  max_score={max_score}, device={device}", flush=True)

    # Load tensor for expert boards and metadata
    print("Loading tensor...", flush=True)
    data = torch.load(args.tensor, weights_only=True, map_location='cpu')
    boards = data['boards']  # (N, 9, 9) int8
    val_targets = data['val_targets']  # (N, 64) two-hot
    turns_remaining = data['turns_remaining']  # (N,) int32
    N = boards.shape[0]

    # Decode val_targets to scalar TD returns
    bins = torch.linspace(0, float(data['max_score']), int(data['num_value_bins']))
    td_returns = (val_targets * bins).sum(dim=-1)

    print(f"  {N:,} states, max_score={data['max_score']}, gamma={data['gamma']}")

    # ================================================================
    print("\n" + "=" * 60, flush=True)
    print("TEST 1: TRAP TEST — Random Chaos Boards", flush=True)
    print("=" * 60, flush=True)
    print("Do random boards get high value predictions?", flush=True)

    from alphatrain.observation import build_observation
    n_chaos = 1000
    n_expert = 1000
    n_blunder = 1000

    # Generate chaos boards: 30-50 random balls on 9x9
    chaos_boards = []
    rng = np.random.RandomState(42)
    for _ in range(n_chaos):
        b = np.zeros((9, 9), dtype=np.int8)
        n_balls = rng.randint(30, 55)
        positions = rng.choice(81, n_balls, replace=False)
        for pos in positions:
            b[pos // 9, pos % 9] = rng.randint(1, 8)
        chaos_boards.append(b)

    # Sample expert boards (from middle of games)
    mid_mask = (turns_remaining > 500) & (turns_remaining < 2000)
    mid_indices = mid_mask.nonzero(as_tuple=True)[0]
    expert_idx = mid_indices[torch.randperm(len(mid_indices))[:n_expert]]
    expert_boards_np = boards[expert_idx].numpy()

    # Generate blunder boards: expert board + 1 random move
    blunder_boards = []
    for i in range(n_blunder):
        b = expert_boards_np[i % n_expert].copy()
        occ = np.argwhere(b != 0)
        emp = np.argwhere(b == 0)
        if len(occ) > 0 and len(emp) > 0:
            src = occ[rng.randint(len(occ))]
            tgt = emp[rng.randint(len(emp))]
            b[tgt[0], tgt[1]] = b[src[0], src[1]]
            b[src[0], src[1]] = 0
        blunder_boards.append(b)

    # Build observations and predict values
    def predict_boards(board_list, label):
        next_r = np.zeros(3, dtype=np.int32)
        next_c = np.zeros(3, dtype=np.int32)
        next_col = np.zeros(3, dtype=np.int32)
        obs_list = []
        for b in board_list:
            obs = build_observation(b, next_r, next_c, next_col, 0)
            obs_list.append(obs)
        obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(device)
        values = []
        with torch.inference_mode():
            for i in range(0, len(obs_t), 64):
                batch = obs_t[i:i+64]
                _, val_logits = net(batch)
                v = net.predict_value(val_logits, max_val=max_score)
                values.append(v.cpu())
        values = torch.cat(values).numpy()
        print(f"\n  {label} ({len(board_list)} boards):", flush=True)
        print(f"    Mean: {values.mean():.1f}", flush=True)
        print(f"    Median: {np.median(values):.1f}", flush=True)
        print(f"    Std: {values.std():.1f}", flush=True)
        print(f"    Min: {values.min():.1f}, Max: {values.max():.1f}", flush=True)
        print(f"    P10: {np.percentile(values, 10):.1f}, "
              f"P90: {np.percentile(values, 90):.1f}", flush=True)
        # How many predict "high" value (above median expert)?
        return values

    expert_vals = predict_boards(list(expert_boards_np), "Expert mid-game boards")
    blunder_vals = predict_boards(blunder_boards, "Blunder boards (expert + 1 random move)")
    chaos_vals = predict_boards(chaos_boards, "Chaos boards (random 30-50 balls)")

    expert_median = np.median(expert_vals)
    print(f"\n  Separation quality (expert median = {expert_median:.1f}):", flush=True)
    print(f"    Expert > median:  {(expert_vals > expert_median).mean()*100:.0f}%", flush=True)
    print(f"    Blunder > median: {(blunder_vals > expert_median).mean()*100:.0f}%", flush=True)
    print(f"    Chaos > median:   {(chaos_vals > expert_median).mean()*100:.0f}%", flush=True)

    # ================================================================
    print("\n" + "=" * 60, flush=True)
    print("TEST 2: TARGET ANALYSIS — TD Returns vs Remaining Turns", flush=True)
    print("=" * 60, flush=True)
    print("Which correlates better with board health?", flush=True)

    # Sample 100K positions for correlation analysis
    sample_idx = torch.randperm(N)[:100000]
    s_boards = boards[sample_idx]
    s_td = td_returns[sample_idx].numpy()
    s_turns = turns_remaining[sample_idx].float().numpy()

    # Compute empty squares for each board
    s_empty = (s_boards == 0).sum(dim=(1, 2)).float().numpy()

    # Correlations
    corr_td_empty = np.corrcoef(s_td, s_empty)[0, 1]
    corr_turns_empty = np.corrcoef(s_turns, s_empty)[0, 1]
    corr_td_turns = np.corrcoef(s_td, s_turns)[0, 1]

    print(f"\n  Correlation with empty squares:", flush=True)
    print(f"    TD returns (gamma=0.95): r = {corr_td_empty:.4f}", flush=True)
    print(f"    Remaining turns:         r = {corr_turns_empty:.4f}", flush=True)
    print(f"    TD vs turns:             r = {corr_td_turns:.4f}", flush=True)

    # Binned analysis: how does each metric vary with empty squares?
    print(f"\n  Value by empty square count:", flush=True)
    print(f"  {'Empty':>6} {'Count':>8} {'TD mean':>8} {'TD std':>8} "
          f"{'Turns mean':>10} {'Turns std':>10}", flush=True)
    for lo in range(10, 71, 10):
        hi = lo + 10
        mask = (s_empty >= lo) & (s_empty < hi)
        n = mask.sum()
        if n > 100:
            print(f"  {lo:>3}-{hi:<3} {n:>8} {s_td[mask].mean():>8.1f} "
                  f"{s_td[mask].std():>8.1f} {s_turns[mask].mean():>10.0f} "
                  f"{s_turns[mask].std():>10.0f}", flush=True)

    # ================================================================
    print("\n" + "=" * 60, flush=True)
    print("TEST 3: VALUE BY GAME STAGE", flush=True)
    print("=" * 60, flush=True)
    print("What does the model predict at different game stages?", flush=True)

    # Sample boards at different stages
    stages = [
        ("Early (>2000 turns left)", turns_remaining > 2000),
        ("Mid (500-2000 turns)", (turns_remaining >= 500) & (turns_remaining <= 2000)),
        ("Late (100-500 turns)", (turns_remaining >= 100) & (turns_remaining < 500)),
        ("Endgame (<100 turns)", turns_remaining < 100),
        ("Death (<20 turns)", turns_remaining < 20),
    ]

    next_r = np.zeros(3, dtype=np.int32)
    next_c = np.zeros(3, dtype=np.int32)
    next_col = np.zeros(3, dtype=np.int32)

    for label, mask in stages:
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) < 100:
            print(f"\n  {label}: too few samples ({len(idx)})", flush=True)
            continue
        sample = idx[torch.randperm(len(idx))[:200]]
        sample_boards = boards[sample].numpy()
        sample_td = td_returns[sample].numpy()
        sample_turns = turns_remaining[sample].numpy()
        sample_empty = (boards[sample] == 0).sum(dim=(1, 2)).float().numpy()

        obs_list = [build_observation(b, next_r, next_c, next_col, 0)
                    for b in sample_boards]
        obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(device)
        with torch.inference_mode():
            _, val_logits = net(obs_t)
            preds = net.predict_value(val_logits, max_val=max_score).cpu().numpy()

        print(f"\n  {label} (n={len(sample)}):", flush=True)
        print(f"    Empty squares: {sample_empty.mean():.0f} "
              f"(range {sample_empty.min():.0f}-{sample_empty.max():.0f})", flush=True)
        print(f"    True TD return: {sample_td.mean():.1f} +/- {sample_td.std():.1f}", flush=True)
        print(f"    NN prediction:  {preds.mean():.1f} +/- {preds.std():.1f}", flush=True)
        print(f"    Prediction error (MAE): {np.abs(preds - sample_td).mean():.1f}", flush=True)
        print(f"    Turns remaining: {sample_turns.mean():.0f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("DIAGNOSIS COMPLETE", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
