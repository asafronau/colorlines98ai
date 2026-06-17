"""Does the recorded root Q carry an improvement signal the Gumbel target can use?

The whole pivot (visits -> Q target) rests on one empirical claim: at the DECISIVE
states the value spread among candidates is large AND argmax(Q) disagrees with
argmax(prior). If Q is saturated everywhere (flat ~2.55), softmax(prior+Q) ~=
softmax(prior) and the Gumbel fix is a no-op. Verify BEFORE building the tensor.

For each sampled state we read the parallel top-k lists (cand_prior=log clean prior,
cand_q=root Q, cand_visits) and compute:
  * q_spread        = max(cand_q) - min(cand_q)         (value head's separation)
  * disagree_qp     = argmax(cand_q) != argmax(cand_prior)
  * disagree_vp     = argmax(cand_visits) != argmax(cand_prior)   (the old target)
  * q_gap           = Q[argmax q] - Q[argmax prior]     (improvement Q claims)
We bucket by q_spread and by within-game position (last-decile = crisis onset for
games that die), and report disagreement rates per bucket. The population we need:
high-spread states where Q's pick != the prior's pick -- those are the corrections.

    PYTHONPATH=. python scripts/diag_q_signal.py
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def iter_states(files, max_states):
    n = 0
    for fi, f in enumerate(files):
        try:
            g = json.load(open(f))
        except Exception:
            continue
        moves = g.get('moves', [])
        M = len(moves)
        died = 'prevention' in f or 'recovery' in f or (
            '_score' in f and int(f.split('_score')[-1].split('.')[0]) < 8000)
        for i, mv in enumerate(moves):
            cq = mv.get('cand_q'); cp = mv.get('cand_prior')
            if not cq or not cp or len(cq) < 2:
                continue
            yield mv, (i + 1) / M, died
            n += 1
            if n >= max_states:
                return
        if (fi + 1) % 200 == 0:
            print(f"  ...scanned {fi+1}/{len(files)} files, {n:,} states", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', nargs='+',
                   default=['data/relabel_v15', 'data/selfplay_v15'])
    p.add_argument('--max-states', type=int, default=300_000)
    p.add_argument('--sample-games', type=int, default=600,
                   help='Cap games per dir (random-ish via stride) to bound runtime.')
    a = p.parse_args()

    files = []
    for d in a.dirs:
        fs = sorted(glob.glob(os.path.join(d, 'game_seed*.json')))
        if a.sample_games and len(fs) > a.sample_games:
            step = len(fs) // a.sample_games
            fs = fs[::step][:a.sample_games]
        files.extend(fs)
        print(f"{d}: {len(fs)} games sampled", flush=True)

    spread, gap, disq, disv, pos, died_f = [], [], [], [], [], []
    for mv, fpos, died in iter_states(files, a.max_states):
        cq = np.asarray(mv['cand_q'], dtype=np.float64)
        cp = np.asarray(mv['cand_prior'], dtype=np.float64)
        cv = np.asarray(mv.get('cand_visits', [0] * len(cq)), dtype=np.float64)
        qa, pa = int(cq.argmax()), int(cp.argmax())
        va = int(cv.argmax())
        spread.append(cq.max() - cq.min())
        gap.append(cq[qa] - cq[pa])
        disq.append(qa != pa)
        disv.append(va != pa)
        pos.append(fpos); died_f.append(died)

    spread = np.array(spread); gap = np.array(gap)
    disq = np.array(disq); disv = np.array(disv)
    pos = np.array(pos); died_f = np.array(died_f)
    N = len(spread)
    print(f"\n=== {N:,} states ===")
    print(f"q_spread (max-min over candidates):")
    for q in [50, 75, 90, 95, 99]:
        print(f"   P{q}: {np.percentile(spread, q):.4f}")
    print(f"   mean {spread.mean():.4f}  max {spread.max():.4f}")
    print(f"\ndisagree(argmax Q, argmax prior):  {100*disq.mean():.1f}%   "
          f"[old target disagree(visits,prior): {100*disv.mean():.1f}%]")

    print(f"\n--- disagreement & gap by q_spread bucket ---")
    edges = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (spread >= lo) & (spread < hi)
        if m.sum() == 0:
            continue
        print(f"  spread [{lo:.2f},{hi:.2f}): {m.sum():>7,} states "
              f"({100*m.mean():4.1f}%) | disagree(Q,prior) {100*disq[m].mean():5.1f}% "
              f"| mean |q_gap| when disagree {np.abs(gap[m & disq]).mean() if (m&disq).any() else 0:.3f}")

    print(f"\n--- by within-game position (last decile ~ crisis onset) ---")
    for lo, hi, lbl in [(0,0.5,'first half'),(0.5,0.9,'mid-late'),(0.9,1.01,'last decile')]:
        m = (pos >= lo) & (pos < hi)
        if m.sum() == 0: continue
        print(f"  {lbl:12s}: {m.sum():>7,} | mean spread {spread[m].mean():.4f} "
              f"| disagree(Q,prior) {100*disq[m].mean():5.1f}%")

    print(f"\n--- died games (crisis corpus) vs capped (survived) ---")
    for val, lbl in [(True,'died/low-score'),(False,'survived/capped')]:
        m = died_f == val
        if m.sum() == 0: continue
        ml = m & (pos >= 0.9)
        print(f"  {lbl:16s}: {m.sum():>7,} states | mean spread {spread[m].mean():.4f} "
              f"| disagree {100*disq[m].mean():4.1f}% || last-decile: "
              f"spread {spread[ml].mean() if ml.any() else 0:.4f} "
              f"disagree {100*disq[ml].mean() if ml.any() else 0:.1f}%")

    # The population that actually matters for the Gumbel target:
    decisive = (spread >= 0.05) & disq
    print(f"\n>>> CORRECTABLE states (spread>=0.05 AND argmax Q != argmax prior): "
          f"{decisive.sum():,} ({100*decisive.mean():.2f}% of all). "
          f"mean |q_gap| {np.abs(gap[decisive]).mean() if decisive.any() else 0:.3f} <<<")
    print("If this population is ~0%, Q is saturated and the Gumbel target degenerates "
          "to the prior. If it's a few %, that's the sparse signal the recipe targets.")


if __name__ == '__main__':
    main()
