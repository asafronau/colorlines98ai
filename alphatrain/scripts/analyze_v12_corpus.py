"""Analyze the V12 corpus to characterize crisis vs self-play data.

Compares the distributions of:
- Visit-target entropy (info density per state — peakier = student gets more signal)
- Board empty count (which "regime" the state lives in)
- Top-1 visit probability (how decisive the teacher is)

Reads JSON game files directly — no tensor build needed. ~3-5 min for V12.

Usage:
    python -m alphatrain.scripts.analyze_v12_corpus \\
        --selfplay-dir data/selfplay_v12 \\
        --crisis-dir data/crisis_v12 \\
        --max-files 2000   # subsample if you want it faster
"""

import os
import json
import glob
import argparse
import time
import numpy as np


def softmax_from_log(log_scores):
    """Recover visit probabilities from top_scores (which are log(visit+eps))."""
    log = np.asarray(log_scores, dtype=np.float64)
    m = log.max()
    exp = np.exp(log - m)
    return exp / exp.sum()


def entropy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def analyze_files(files, source_label, sample_states_per_game=None):
    """Walk JSON files, extract per-state metrics."""
    rec = {
        'empty': [],          # empty squares per state
        'entropy': [],        # visit-distribution entropy
        'top1': [],           # P(top-1 move) per state
        'top3': [],           # P(top-3 cum)
        'n_top': [],          # how many moves have positive visits
    }
    n_games = 0
    n_states = 0
    t0 = time.time()

    for fi, f in enumerate(files):
        try:
            with open(f) as fp:
                game = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        n_games += 1
        moves = game.get('moves', [])
        if sample_states_per_game and len(moves) > sample_states_per_game:
            idx = np.linspace(0, len(moves) - 1,
                              sample_states_per_game).astype(int)
            moves_iter = [moves[i] for i in idx]
        else:
            moves_iter = moves

        for m in moves_iter:
            board = np.asarray(m['board'], dtype=np.int8)
            rec['empty'].append(int((board == 0).sum()))
            scores = m.get('top_scores', [])
            if not scores:
                continue
            probs = softmax_from_log(scores)
            rec['entropy'].append(entropy(probs))
            rec['top1'].append(float(probs[0]))
            rec['top3'].append(float(probs[:3].sum()))
            rec['n_top'].append(int(len(probs)))
            n_states += 1

        if (fi + 1) % 1000 == 0:
            print(f"  {source_label}: {fi+1}/{len(files)} files, "
                  f"{n_states:,} states ({time.time()-t0:.0f}s)",
                  flush=True)

    return rec, n_games, n_states


def summarize(rec, label):
    print(f"\n=== {label} ===")
    for key, arr in rec.items():
        if not arr:
            continue
        a = np.asarray(arr, dtype=np.float64)
        print(f"  {key:>8}: mean={a.mean():7.3f}  "
              f"P10={np.percentile(a,10):7.3f}  "
              f"P50={np.percentile(a,50):7.3f}  "
              f"P90={np.percentile(a,90):7.3f}  "
              f"min={a.min():.3f}  max={a.max():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selfplay-dir', default='data/selfplay_v12')
    p.add_argument('--crisis-dir', default='data/crisis_v12')
    p.add_argument('--max-files', type=int, default=0,
                   help='Subsample N files per source (0=all)')
    p.add_argument('--sample-states-per-game', type=int, default=0,
                   help='Subsample N states per game (0=all)')
    args = p.parse_args()

    sample = args.sample_states_per_game or None

    print(f"Selfplay: {args.selfplay_dir}")
    sp_files = sorted(glob.glob(os.path.join(args.selfplay_dir, '*.json')))
    print(f"  {len(sp_files)} files")
    if args.max_files:
        sp_files = sp_files[:args.max_files]
        print(f"  using {len(sp_files)} (subsampled)")

    print(f"Crisis: {args.crisis_dir}")
    cr_files = sorted(glob.glob(os.path.join(args.crisis_dir, '*.json')))
    cr_recovery = [f for f in cr_files if '_recovery_' in f]
    cr_prevention = [f for f in cr_files if '_prevention_' in f]
    print(f"  recovery: {len(cr_recovery)} files, "
          f"prevention: {len(cr_prevention)} files")
    if args.max_files:
        cr_recovery = cr_recovery[:args.max_files]
        cr_prevention = cr_prevention[:args.max_files]
        print(f"  using {len(cr_recovery)} + {len(cr_prevention)} (subsampled)")

    sp_rec, sp_g, sp_s = analyze_files(sp_files, 'selfplay', sample)
    print(f"selfplay: {sp_g} games, {sp_s:,} states")

    rec_rec, rec_g, rec_s = analyze_files(cr_recovery, 'recovery', sample)
    print(f"recovery: {rec_g} games, {rec_s:,} states")

    pre_rec, pre_g, pre_s = analyze_files(cr_prevention, 'prevention', sample)
    print(f"prevention: {pre_g} games, {pre_s:,} states")

    summarize(sp_rec, 'SELFPLAY')
    summarize(rec_rec, 'CRISIS RECOVERY')
    summarize(pre_rec, 'CRISIS PREVENTION')

    # Side-by-side key metrics
    print("\n=== SIDE-BY-SIDE (P50) ===")
    print(f"  {'metric':>15}  {'selfplay':>10}  {'recovery':>10}  {'prevention':>10}")
    for key in ['empty', 'entropy', 'top1', 'top3']:
        sp_v = np.percentile(sp_rec[key], 50) if sp_rec[key] else 0
        rec_v = np.percentile(rec_rec[key], 50) if rec_rec[key] else 0
        pre_v = np.percentile(pre_rec[key], 50) if pre_rec[key] else 0
        print(f"  {key:>15}  {sp_v:>10.3f}  {rec_v:>10.3f}  {pre_v:>10.3f}")

    # State volume
    total = sp_s + rec_s + pre_s
    print(f"\n=== STATE COUNTS ===")
    print(f"  selfplay:   {sp_s:>10,}  ({100*sp_s/total:.1f}%)")
    print(f"  recovery:   {rec_s:>10,}  ({100*rec_s/total:.1f}%)")
    print(f"  prevention: {pre_s:>10,}  ({100*pre_s/total:.1f}%)")
    print(f"  total:      {total:>10,}")


if __name__ == '__main__':
    main()
