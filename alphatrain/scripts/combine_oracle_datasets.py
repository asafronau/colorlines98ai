"""Combine multiple Phase 1 oracle datasets into one, with deep inspection.

Loads N files (phase1_oracle.pt format), de-duplicates by anchor board hash
to avoid double-counting any anchor that happened to be sampled across runs,
re-IDs anchors with a contiguous range, and writes a combined .pt.

Also prints a detailed inspection of the combined corpus for review:
  - per-source breakdown
  - margin distribution (histogram of Δcap_rate when oracle disagrees)
  - disagreement type breakdown (top1 wrong vs lower-rank wins)
  - corpus diversity (turn distribution, board fullness)
  - Δcap_rate vs Δmean_turns correlation

Usage:
    python -m alphatrain.scripts.combine_oracle_datasets \\
        --inputs alphatrain/data/phase1_oracle.pt \\
                  alphatrain/data/phase1_oracle_v2_crisis5k.pt \\
                  alphatrain/data/phase1_oracle_fleet_10000.pt \\
        --output alphatrain/data/phase1_oracle_combined.pt
"""

import argparse
import hashlib
import os
from collections import Counter, defaultdict
import numpy as np
import torch


def board_hash(board):
    return hashlib.md5(np.asarray(board, dtype=np.int8).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--margin-threshold', type=float, default=0.15)
    args = p.parse_args()

    all_results = []
    by_source = {}
    seen_hashes = set()
    n_dup = 0
    for fn in args.inputs:
        data = torch.load(fn, weights_only=False)
        results = data['results']
        kept = 0
        for r in results:
            h = board_hash(r['anchor_board'])
            if h in seen_hashes:
                n_dup += 1
                continue
            seen_hashes.add(h)
            r = dict(r)  # shallow copy
            r['source_file'] = os.path.basename(fn)
            r['original_anchor_id'] = r['anchor_id']
            r['anchor_id'] = len(all_results)
            all_results.append(r)
            kept += 1
        by_source[os.path.basename(fn)] = kept
        print(f"  loaded {os.path.basename(fn)}: {kept} kept (of {len(results)})",
              flush=True)
    if n_dup:
        print(f"  dropped {n_dup} duplicates (same anchor_board hash)",
              flush=True)
    print(f"  total combined: {len(all_results)} unique anchors", flush=True)

    out = {
        'args': vars(args),
        'results': all_results,
        'sources': by_source,
    }
    torch.save(out, args.output)
    print(f"\nSaved {args.output} "
          f"({os.path.getsize(args.output)/1e6:.0f} MB)", flush=True)

    # ── DEEP INSPECTION ──
    print(f"\n{'='*64}\nDEEP INSPECTION\n{'='*64}", flush=True)

    by_label = Counter()
    n_disagree = 0
    n_high_margin = 0
    margins_cap = []
    margins_turns = []
    margins_score = []
    rank_of_winner = Counter()
    n_eligible = 0
    n_singleton = 0
    turn_dist = []
    ball_count_dist = []

    # For correlation
    paired_margins = []

    for r in all_results:
        pm = r['per_move']
        if len(pm) < 2:
            n_singleton += 1
            continue
        n_eligible += 1
        by_label[r.get('source_label', 'unknown')] += 1
        turn_dist.append(r.get('turn_origin', 0))
        ball_count_dist.append(int((np.asarray(r['anchor_board']) != 0).sum()))

        sorted_mv = sorted(pm.items(), key=lambda kv: kv[1]['rank'])
        policy_top1 = sorted_mv[0][1]
        # Find the move that maximizes cap_rate
        best_move, best_data = max(pm.items(), key=lambda kv: kv[1]['cap_rate'])
        margin = best_data['cap_rate'] - policy_top1['cap_rate']
        if margin > 0:
            n_disagree += 1
            margins_cap.append(margin)
            d_turns = best_data['mean_turns'] - policy_top1['mean_turns']
            d_score = best_data['mean_score'] - policy_top1['mean_score']
            margins_turns.append(d_turns)
            margins_score.append(d_score)
            paired_margins.append((margin, d_turns, d_score))
            rank_of_winner[best_data['rank']] += 1
            if margin >= args.margin_threshold:
                n_high_margin += 1

    print(f"\nEligible anchors (≥2 moves): {n_eligible}  "
          f"(skipped {n_singleton} singletons)", flush=True)

    print(f"\nSource label distribution:", flush=True)
    for k, v in by_label.most_common():
        print(f"  {k}: {v}  ({100*v/n_eligible:.1f}%)", flush=True)

    print(f"\nSource file distribution:", flush=True)
    file_counts = Counter(r['source_file'] for r in all_results)
    for k, v in file_counts.most_common():
        print(f"  {k}: {v}", flush=True)

    print(f"\nDisagreement: {n_disagree}/{n_eligible} = "
          f"{100*n_disagree/n_eligible:.1f}%", flush=True)
    print(f"High-margin (Δcap_rate ≥ {args.margin_threshold}): "
          f"{n_high_margin} ({100*n_high_margin/n_eligible:.1f}% of eligible)",
          flush=True)

    if margins_cap:
        arr = np.array(margins_cap)
        print(f"\nΔcap_rate distribution (when oracle wins):", flush=True)
        print(f"  count: {len(arr)}", flush=True)
        print(f"  mean: {arr.mean():.4f}  median: {np.median(arr):.4f}",
              flush=True)
        print(f"  P10: {np.percentile(arr, 10):.4f}  "
              f"P25: {np.percentile(arr, 25):.4f}  "
              f"P50: {np.percentile(arr, 50):.4f}  "
              f"P75: {np.percentile(arr, 75):.4f}  "
              f"P90: {np.percentile(arr, 90):.4f}  "
              f"P95: {np.percentile(arr, 95):.4f}", flush=True)
        # Buckets
        print(f"\n  bucket counts:", flush=True)
        for lo, hi in [(0, 0.05), (0.05, 0.10), (0.10, 0.15),
                        (0.15, 0.25), (0.25, 0.50), (0.50, 1.01)]:
            n_b = ((arr >= lo) & (arr < hi)).sum()
            print(f"    [{lo:.2f}, {hi:.2f}): {n_b} "
                  f"({100*n_b/len(arr):.1f}%)", flush=True)

    if margins_turns:
        arr = np.array(margins_turns)
        print(f"\nΔmean_turns distribution (when oracle wins by cap_rate):",
              flush=True)
        print(f"  mean: {arr.mean():.1f}  median: {np.median(arr):.1f}",
              flush=True)

    if rank_of_winner:
        print(f"\nRank of oracle-best move (in policy's ranking):", flush=True)
        for rank in sorted(rank_of_winner):
            n = rank_of_winner[rank]
            print(f"  rank {rank}: {n}  ({100*n/n_disagree:.1f}%)", flush=True)

    if paired_margins:
        arr = np.array(paired_margins)
        # Correlation between cap_rate margin and turns margin
        cap_arr = arr[:, 0]
        turns_arr = arr[:, 1]
        if cap_arr.std() > 0 and turns_arr.std() > 0:
            corr = np.corrcoef(cap_arr, turns_arr)[0, 1]
            print(f"\nCorrelation (Δcap_rate, Δmean_turns): {corr:.3f}",
                  flush=True)

    # Diversity stats
    if turn_dist:
        arr = np.array(turn_dist)
        print(f"\nAnchor turn_origin (when anchor was sampled):", flush=True)
        print(f"  mean: {arr.mean():.0f}  median: {np.median(arr):.0f}  "
              f"P10: {np.percentile(arr, 10):.0f}  "
              f"P90: {np.percentile(arr, 90):.0f}", flush=True)
    if ball_count_dist:
        arr = np.array(ball_count_dist)
        print(f"\nBall count at anchor (board fullness):", flush=True)
        print(f"  mean: {arr.mean():.1f}  median: {np.median(arr):.0f}  "
              f"P10: {np.percentile(arr, 10):.0f}  "
              f"P90: {np.percentile(arr, 90):.0f}", flush=True)
        # Buckets
        print(f"  bucket counts:", flush=True)
        for lo, hi in [(0, 20), (20, 40), (40, 50), (50, 60),
                        (60, 70), (70, 82)]:
            n_b = ((arr >= lo) & (arr < hi)).sum()
            print(f"    [{lo}, {hi}): {n_b} ({100*n_b/len(arr):.1f}%)",
                  flush=True)


if __name__ == '__main__':
    main()
