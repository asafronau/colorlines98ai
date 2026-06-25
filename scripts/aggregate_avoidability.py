"""Aggregate rewind_from_death outputs into the AVOIDABILITY-vs-OPENNESS curve.

The quiet-play test (does stable play accumulate fixable inaccuracies, or is the
floor only crisis/RNG?). Each rewind_from_death output records, per rewind depth, a
decision point with its board `empties` (openness) and a held-out-verified verdict.

A REAL fork = flag==True AND its held-out bootstrap CI excludes 0 (survives==True):
a DIFFERENT policy-candidate move that robustly lowers death-within-H on FRESH seeds.

We bin REAL forks by `empties` at the fork:
  - REAL forks at LOW empties (crowded, <=20)  -> CRISIS mistakes (known lever).
  - REAL forks at HIGH empties (open, >=35)     -> QUIET-PLAY inaccuracy: the policy
        made a fixable mistake while the board still looked fine = early accumulation.
        These are mineable as targeted corrections (board+next_balls+cand_rates).

`sealed` rows (every candidate dies >60%) are board-doomed regardless of move -> not a
decision the policy could have won; reported separately, excluded from the denominator.

    PYTHONPATH=. python scripts/aggregate_avoidability.py '/tmp/rewind_sweep/*.json'
"""
import sys
import glob
import json

BUCKETS = [(0, 15, '<=15 crowded'), (16, 25, '16-25'), (26, 35, '26-35'),
           (36, 45, '36-45 open'), (46, 81, '>=46 very open')]


def bucket(e):
    for lo, hi, label in BUCKETS:
        if e is None:
            return 'unknown'
        if lo <= e <= hi:
            return label
    return 'unknown'


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rewind_sweep/*.json'
    files = sorted(glob.glob(pat))
    if not files:
        print(f"No files match {pat}")
        return
    # per bucket: [n_decision, n_sealed, n_flagged, n_real, sum_real_gap]
    agg = {label: [0, 0, 0, 0, 0.0] for _, _, label in BUCKETS}
    agg['unknown'] = [0, 0, 0, 0, 0.0]
    real_examples = []
    n_games = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  skip {f}: {e}"); continue
        n_games += 1
        held = d.get('held', {})
        for r in d.get('depth_rows', []):
            b = bucket(r.get('empties'))
            cell = agg[b]
            if r.get('sealed'):
                cell[1] += 1
                continue
            cell[0] += 1                                   # a winnable decision point
            if r.get('flag'):
                cell[2] += 1
                h = held.get(f"{r['di']}_{r['best_mi']}")
                if h and h.get('survives'):
                    cell[3] += 1
                    cell[4] += h.get('gap', 0.0)
                    real_examples.append({
                        'game': d.get('meta', {}).get('seed'), 'turn': r.get('turn'),
                        'empties': r.get('empties'), 'gap': round(h.get('gap', 0.0), 1),
                        'ci': [round(h.get('lo', 0), 1), round(h.get('hi', 0), 1)],
                        'pol_move': r.get('pol_move'), 'best_move': r.get('best_move')})

    print(f"\n{n_games} games | AVOIDABILITY vs BOARD OPENNESS (empties at the fork)\n")
    hdr = f"{'openness bucket':>16} {'decisions':>10} {'sealed':>7} {'flagged':>8} {'REAL':>5} {'avoid%':>7} {'meanΔ':>7}"
    print(hdr); print('-' * len(hdr))
    tot = [0, 0, 0, 0, 0.0]
    for _, _, label in BUCKETS:
        c = agg[label]
        for i in range(5):
            tot[i] += c[i]
        avoid = 100.0 * c[3] / c[0] if c[0] else 0.0
        mg = c[4] / c[3] if c[3] else 0.0
        print(f"{label:>16} {c[0]:>10} {c[1]:>7} {c[2]:>8} {c[3]:>5} {avoid:>6.1f}% {mg:>6.1f}")
    u = agg['unknown']
    if u[0] or u[1]:
        print(f"{'unknown':>16} {u[0]:>10} {u[1]:>7} {u[2]:>8} {u[3]:>5}")
    avoid = 100.0 * tot[3] / tot[0] if tot[0] else 0.0
    print('-' * len(hdr))
    print(f"{'TOTAL':>16} {tot[0]:>10} {'':>7} {tot[2]:>8} {tot[3]:>5} {avoid:>6.1f}%")

    print("\n--- READ ---")
    print("REAL forks at HIGH empties (>=36 open) => quiet-play inaccuracy is real & mineable")
    print("                                          (targeted corrections, NOT distillation).")
    print("REAL forks ONLY at LOW empties (<=20)  => quiet play near-optimal; floor is crisis/RNG.")

    if real_examples:
        real_examples.sort(key=lambda x: -(x['empties'] or 0))
        print(f"\nTop REAL forks by openness (highest empties first; the quiet-play candidates):")
        for ex in real_examples[:15]:
            print(f"  empties {ex['empties']:>2} turn {ex['turn']:>4} game {ex['game']} "
                  f"Δcat {ex['gap']:>5}pp CI{ex['ci']}  pol{ex['pol_move']} -> best{ex['best_move']}")


if __name__ == '__main__':
    main()
