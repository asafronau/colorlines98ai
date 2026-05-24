"""Classify the failure mode of each trajectory in logs/instrumented_failures/.

Buckets:
  A_early_rng:    final_turn < 30
  B_density_spiral: empties trend monotonically down ≥20 turns ending in death
  C_no_clears:    long stretch with NO line clears before death (≥50 turns)
  D_slow_drift:   survived ≥100 turns but score/turn < 1.5
  R_recovered:    final_score >= 1000 (not a failure in this run)
  U_unbucketed
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import Counter, defaultdict

import numpy as np


def classify(g):
    m = g['metrics']
    final_score = g['final_score']
    final_turn = g['final_turn']
    if final_score >= 1000:
        return 'R_recovered'
    if not m or final_turn < 30:
        return 'A_early_rng'

    # Density spiral check
    if len(m) >= 20:
        last20 = m[-20:]
        emp = [x['empties'] for x in last20]
        decs = sum(1 for i in range(1, len(emp)) if emp[i] < emp[i - 1])
        incs = sum(1 for i in range(1, len(emp)) if emp[i] > emp[i - 1])
        if decs >= incs * 2 and emp[0] - emp[-1] >= 5:
            return 'B_density_spiral'

    # No-clears stretch: ≥50 consecutive turns without a 'cleared' annotation
    if len(m) >= 50:
        clear_turns = [i for i, x in enumerate(m) if x.get('cleared', 0) > 0]
        if not clear_turns:
            return 'C_no_clears'
        last_clear = clear_turns[-1]
        if (len(m) - last_clear) >= 50:
            return 'C_no_clears'

    # Slow drift
    if final_turn >= 100 and final_score / max(1, final_turn) < 1.5:
        return 'D_slow_drift'

    return 'U_unbucketed'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='logs/instrumented_failures')
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, 'traj_seed*.json')))
    print(f"Loaded {len(files)} trajectories")
    by_bucket = defaultdict(list)
    for f in files:
        with open(f) as fp:
            g = json.load(fp)
        b = classify(g)
        by_bucket[b].append(g)

    print(f"\n{'bucket':<20s} | {'count':>5s} | {'med score':>9s} | "
          f"{'med turn':>8s} | {'med empties at end':>18s} | seeds")
    print('-' * 100)
    order = ['A_early_rng', 'B_density_spiral', 'C_no_clears',
             'D_slow_drift', 'U_unbucketed', 'R_recovered']
    for bucket in order:
        examples = by_bucket.get(bucket, [])
        if not examples:
            continue
        scores = [g['final_score'] for g in examples]
        turns = [g['final_turn'] for g in examples]
        # Median empties at last turn
        end_empties = [g['metrics'][-1]['empties']
                        if g['metrics'] else 81
                        for g in examples]
        sample_seeds = [str(g['seed']) for g in
                         sorted(examples, key=lambda g: g['final_score'])][:5]
        print(f"  {bucket:<18s} | {len(examples):>5d} | "
              f"{int(np.median(scores)):>9d} | "
              f"{int(np.median(turns)):>8d} | "
              f"{int(np.median(end_empties)):>18d} | "
              f"{', '.join(sample_seeds)}{'...' if len(examples) > 5 else ''}")

    # Detailed per-game for true failures, including clear rate
    # Equilibrium clear rate to keep up with 3-ball/turn spawns: 3/8 = 37.5%
    EQUILIBRIUM = 3.0 / 8.0
    print(f"\n=== Per-game detail (failures only) ===")
    print(f"   Equilibrium clearing rate = 3/8 = {EQUILIBRIUM*100:.1f}% "
          f"(5-clear removes 5 balls + skips 3-ball spawn)")
    print()
    print(f"  {'seed':<8s} score | turns | clears | clear_rate | "
          f"pts/turn | end empties | game_over")
    for bucket in ['B_density_spiral', 'C_no_clears',
                    'A_early_rng', 'D_slow_drift', 'U_unbucketed']:
        examples = sorted(by_bucket.get(bucket, []),
                           key=lambda g: g['final_score'])
        for g in examples:
            m = g['metrics']
            n_clears = sum(1 for x in m if x.get('cleared', 0) > 0)
            empties_end = m[-1]['empties'] if m else 81
            turn = g['final_turn']
            score = g['final_score']
            clear_rate = n_clears / max(1, turn)
            below = "↓BELOW" if clear_rate < EQUILIBRIUM else "↑above"
            print(f"  {g['seed']:<8d} {score:>5d}  | {turn:>5d} | "
                  f"{n_clears:>6d} | {clear_rate*100:>5.1f}% {below} | "
                  f"{score/max(1, turn):>6.2f}   | "
                  f"{empties_end:>10d}  | {g['game_over']}")


if __name__ == '__main__':
    main()
