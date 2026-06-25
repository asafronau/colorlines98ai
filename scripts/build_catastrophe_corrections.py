"""Convert the rollout catastrophe-fork harvest -> the PROVEN soft-corrections corpus.

The harvest (overnight_systematic -> mine_crisis_sweep) produces, per CONFIRMED fork
(R=500 paired-bootstrap CI excludes 0), a set of candidate moves each with a rollout
catastrophe rate (P(died within horizon)). This is the action-risk teacher signal.

pillar3f was made by TASK ARITHMETIC on a soft-corrections corpus (HISTORY 167-168:
pillar3f = pillar3b + 0.5*decisive_vector). The MERGE channel is robust to corpus noise
where the listwise --aux channel (pillar3c) regressed ("the filter was compensating for
the channel, not the data"). So we route these forks through the SAME proven channel by
emitting the soft-corrections format `train_crisis_ft.py` consumes (tgt_idx/tgt_prob/
weight/seed) -- with the target derived from catastrophe instead of MCTS visits:

    tgt_prob_i  proportional to exp(-catastrophe_i / T)   over the fork's candidate moves
                (lower catastrophe -> more mass; the confirmed-safer move dominates)
    weight      = confirmed catastrophe gap (pp), normalized to mean 1 (anti-dilution)

Then: train_crisis_ft.py --corpus <this> --base pillar3f  (frozen BN, fit the corrections)
  ->  merge_checkpoints.py --base pillar3f --crisis <ft> --alpha {sweep}  (task vector)
  ->  gate: scripts/normal_play_drift.py (no normal-play forgetting) + full-game eval.

    PYTHONPATH=. python scripts/build_catastrophe_corrections.py \\
        --mine-glob 'logs/harvest2/mine_*.json' --temp 10 \\
        --out alphatrain/data/catastrophe_corrections_pillar3f_h2.pt
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.counterfactual import _move_index_from_list

K = 20   # max candidate moves stored per anchor (cand_rates is pol-k + fv-k ~ 22)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mine-glob', default='logs/harvest2/mine_*.json')
    p.add_argument('--temp', type=float, default=10.0,
                   help='Softmax temperature on -catastrophe (pp). Lower = peakier on the '
                        'safest move. 10 spreads ~5-30pp gaps into a graded preference.')
    p.add_argument('--min-gap', type=float, default=0.0,
                   help='Drop confirmed forks whose gap < this (pp). 0 = keep all confirmed.')
    p.add_argument('--all-flagged', action='store_true',
                   help='Include flagged-but-unconfirmed forks too (merge channel is noise-'
                        'robust; default keeps only CI-confirmed for cleanliness).')
    p.add_argument('--all-rows', action='store_true',
                   help='USE THEM ALL: every band state with a safer candidate (gap>=min_gap), '
                        'not just flagged. Matches the proven corpus scope (build_corrections_'
                        'corpus min_margin=0, weighted). gap-weight makes near-0-gap states inert.')
    p.add_argument('--out', default='alphatrain/data/catastrophe_corrections_pillar3f_h2.pt')
    a = p.parse_args()

    files = sorted(glob.glob(a.mine_glob))
    if not files:
        raise FileNotFoundError(f"no mine files match {a.mine_glob}")

    boards, npos, ncol, nnext = [], [], [], []
    tgt_idx, tgt_prob, weights, seeds = [], [], [], []
    n_files = n_conf = n_flag = n_skip_gap = n_degenerate = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        n_files += 1
        seed = int(d['meta']['seed'])
        confirms = d.get('confirms', {})
        for r in d['rows']:
            flagged = bool(r['flag'])
            if not flagged and not a.all_rows:
                continue
            if flagged:
                n_flag += 1
            c = confirms.get(str(r['depth']))
            confirmed = bool(c and c.get('real'))
            if not a.all_rows and not confirmed and not a.all_flagged:
                continue
            # gap: prefer the clean R=500 re-eval (confirms) where it exists (flagged
            # depths); else the screen gap. R=500-negative flagged forks (winner's-curse
            # false positives) drop out via min_gap=0.
            gap = float(c['gap']) if c else float(r.get('gap', 0.0))
            if gap < a.min_gap:
                n_skip_gap += 1
                continue
            # candidate moves + catastrophe rates at this fork
            cr = r['cand_rates']                       # [[move, cat%, n], ...]
            if len(cr) < 2:
                n_degenerate += 1
                continue
            idxs = [_move_index_from_list(mv) for mv, _cat, _n in cr][:K]
            cats = np.array([float(cat) for _mv, cat, _n in cr][:K], dtype=np.float64)
            # soft target: prefer LOW catastrophe -> softmax(-cat / T) over candidates
            logits = -cats / max(a.temp, 1e-6)
            prs = np.exp(logits - logits.max())
            prs = prs / prs.sum()
            pad = K - len(idxs)

            nposs, ncols = [[0, 0], [0, 0], [0, 0]], [0, 0, 0]
            for i, nb in enumerate(r['next_balls'][:3]):
                (nr, ncc), color = nb
                nposs[i] = [int(nr), int(ncc)]
                ncols[i] = int(color)

            boards.append(r['board'])
            npos.append(nposs); ncol.append(ncols)
            nnext.append(min(len(r['next_balls']), 3))
            tgt_idx.append(idxs + [0] * pad)
            tgt_prob.append(prs.tolist() + [0.0] * pad)
            weights.append(gap)
            seeds.append(seed)
            n_conf += 1 if confirmed else 0

    if not boards:
        raise SystemExit(f"No usable forks in {n_files} files.")
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.mean().clamp(min=1e-6)               # normalize mean -> 1
    corpus = {
        'boards': torch.tensor(boards, dtype=torch.int8),
        'next_pos': torch.tensor(npos, dtype=torch.int8),
        'next_col': torch.tensor(ncol, dtype=torch.int8),
        'n_next': torch.tensor(nnext, dtype=torch.int8),
        'tgt_idx': torch.tensor(tgt_idx, dtype=torch.long),
        'tgt_prob': torch.tensor(tgt_prob, dtype=torch.float32),
        'weight': w,
        'seed': torch.tensor(seeds, dtype=torch.long),
        '_stats': {'n_files': n_files, 'n_anchors': len(boards),
                   'n_confirmed': n_conf, 'n_flagged': n_flag,
                   'n_seeds': len(set(seeds)), 'temp': a.temp,
                   'min_gap': a.min_gap, 'all_flagged': a.all_flagged,
                   'min_margin': a.min_gap},   # alias for train_crisis_ft's printout
    }
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    torch.save(corpus, a.out)
    wn = w.numpy()
    print(f"Wrote {a.out} ({os.path.getsize(a.out)/1e6:.2f} MB)")
    print(f"  {len(boards)} anchors ({n_conf} CI-confirmed) from {len(set(seeds))} seeds "
          f"/ {n_files} games  | temp={a.temp} min_gap={a.min_gap}")
    print(f"  weight (gap pp, mean 1): P10={np.percentile(wn,10):.2f} "
          f"P50={np.percentile(wn,50):.2f} P90={np.percentile(wn,90):.2f} max={wn.max():.1f}")
    # report target peakedness (the safest-move share) for sanity
    tp = corpus['tgt_prob'].numpy()
    top = tp.max(1)
    print(f"  target top-share (safest move mass): P10={np.percentile(top,10):.2f} "
          f"P50={np.percentile(top,50):.2f} P90={np.percentile(top,90):.2f}")


if __name__ == '__main__':
    main()
