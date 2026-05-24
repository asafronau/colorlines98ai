"""Compare 400-sim vs 800-sim crisis runs (same 50 seeds) on continue-turns survived.

Same-seed paired comparison. For each label (recovery/prevention), match
crisis JSONs by original_seed and report paired differences in:
  - continue_turns_survived = turns - replay_from_turn
  - capped@500 rate
  - score gained during the replay

Question: does doubling MCTS sims from 400 → 800 measurably help the model
recover/prevent dying? Answer informs whether V14 selfplay should use 400 or 800.
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict

import numpy as np


def load_run(directory, label):
    """Return dict {original_seed: (turns_survived, capped, score)}."""
    out = {}
    pattern = os.path.join(directory, f'game_seed*_{label}_*.json')
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            g = json.load(f)
        seed = g['original_seed']
        survived = g['turns'] - g['replay_from_turn']
        out[seed] = {
            'turns_survived': int(survived),
            'capped': bool(g.get('capped', False)),
            'score': int(g.get('score', 0)),
            'replay_from_turn': int(g.get('replay_from_turn', 0)),
        }
    return out


def describe(name, values):
    if not values:
        print(f"  {name}: n=0")
        return
    a = np.array(values, dtype=np.float64)
    print(f"  {name}: n={len(a)}  "
          f"mean={a.mean():.0f}  median={np.median(a):.0f}  "
          f"P10={np.percentile(a, 10):.0f}  P25={np.percentile(a, 25):.0f}  "
          f"P75={np.percentile(a, 75):.0f}  max={a.max():.0f}")


def compare(label, r400, r800):
    print(f"\n{'=' * 72}")
    print(f"  {label.upper()} — n_400={len(r400)}, n_800={len(r800)}")
    print(f"{'=' * 72}")

    # Unmatched aggregate
    s400 = [r['turns_survived'] for r in r400.values()]
    s800 = [r['turns_survived'] for r in r800.values()]
    cap_400 = sum(r['capped'] for r in r400.values())
    cap_800 = sum(r['capped'] for r in r800.values())
    score_400 = [r['score'] for r in r400.values()]
    score_800 = [r['score'] for r in r800.values()]
    print(f"\n  Continue-turns survived (post-rewind):")
    describe("    400 sims", s400)
    describe("    800 sims", s800)
    print(f"\n  Capped@500 (didn't die in 500 turns):")
    print(f"    400 sims: {cap_400}/{len(r400)} ({100*cap_400/max(1,len(r400)):.1f}%)")
    print(f"    800 sims: {cap_800}/{len(r800)} ({100*cap_800/max(1,len(r800)):.1f}%)")
    print(f"\n  Score during replay (lifetime score at end):")
    describe("    400 sims", score_400)
    describe("    800 sims", score_800)

    # Paired analysis on common seeds
    common = sorted(set(r400.keys()) & set(r800.keys()))
    if not common:
        print(f"\n  (no paired data)")
        return
    diffs_turns = []
    diffs_score = []
    wins_800, ties, wins_400 = 0, 0, 0
    cap_800_only = 0
    cap_400_only = 0
    cap_both = 0
    cap_neither = 0
    for s in common:
        a = r400[s]
        b = r800[s]
        d_t = b['turns_survived'] - a['turns_survived']
        d_s = b['score'] - a['score']
        diffs_turns.append(d_t)
        diffs_score.append(d_s)
        if d_t > 0:
            wins_800 += 1
        elif d_t < 0:
            wins_400 += 1
        else:
            ties += 1
        ca, cb = a['capped'], b['capped']
        if ca and cb: cap_both += 1
        elif cb and not ca: cap_800_only += 1
        elif ca and not cb: cap_400_only += 1
        else: cap_neither += 1
    print(f"\n  PAIRED (n={len(common)}):")
    print(f"    Δ continue-turns (800 − 400):")
    print(f"      mean={np.mean(diffs_turns):+.1f}  "
          f"median={np.median(diffs_turns):+.0f}  "
          f"std={np.std(diffs_turns, ddof=1):.0f}  "
          f"SE={np.std(diffs_turns, ddof=1)/np.sqrt(len(diffs_turns)):.1f}")
    print(f"      800 wins: {wins_800}/{len(common)} ({100*wins_800/len(common):.1f}%)  "
          f"ties: {ties}  400 wins: {wins_400}/{len(common)} ({100*wins_400/len(common):.1f}%)")
    print(f"    Δ replay score (800 − 400):")
    print(f"      mean={np.mean(diffs_score):+.1f}  "
          f"median={np.median(diffs_score):+.0f}  "
          f"std={np.std(diffs_score, ddof=1):.0f}")
    print(f"    Capping pattern:")
    print(f"      both capped (both survived 500t): {cap_both}")
    print(f"      800 only capped (800 saved a game 400 lost): {cap_800_only}")
    print(f"      400 only capped (400 saved a game 800 lost): {cap_400_only}")
    print(f"      neither capped (both died <500t): {cap_neither}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s400-dir', default='data/crisis_v14_s400')
    parser.add_argument('--s800-dir', default='data/crisis_v14_s800')
    args = parser.parse_args()

    for label in ['recovery', 'prevention']:
        r400 = load_run(args.s400_dir, label)
        r800 = load_run(args.s800_dir, label)
        compare(label, r400, r800)


if __name__ == '__main__':
    main()
