"""Are the MCTS corrections actually BETTER MOVES for pillar3f? (the decisive diagnostic)

The +37% used MCTS@4800 corrections on pillar3b. Re-running the recipe on pillar3f
DEGRADED it (HISTORY 172). Two hypotheses: (1) the corrections aren't genuine improvements
for the now-strong pillar3f (FV-MCTS too weak a teacher), or (2) they ARE genuine but
distilling them causes collateral damage. This separates them.

For a sample of corrections (state S, MCTS-top move A != pillar3f move B), play pillar3f
GREEDILY from each move over R common-RNG rollouts to death; compare mean turns survived and
catastrophe rate (died within horizon). If A robustly out-survives B -> corrections are
genuine (-> the bug is in distillation). If A does NOT beat B -> the teacher is wrong for
pillar3f (-> need a stronger value head, no paradox).

    PYTHONPATH=. python scripts/verify_corrections.py --n 50 --R 64 --horizon 800
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.evaluate import load_model
from scripts.batched_rollout import batched_rollout, _decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3f.pt')
    p.add_argument('--glob', default='crisis/corrections_pillar3f/corr_*.json')
    p.add_argument('--n', type=int, default=50, help='corrections to test')
    p.add_argument('--R', type=int, default=64, help='common-RNG rollouts per move')
    p.add_argument('--horizon', type=int, default=800)
    p.add_argument('--min-margin', type=float, default=0.10,
                   help='only test DECISIVE corrections (mcts_top_share - pol_share >= this)')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--device', default='mps')
    p.add_argument('--seed-base', type=int, default=909000)
    a = p.parse_args()

    dev = torch.device(a.device)
    net, _ = load_model(a.model, dev, fp16=True)
    dtype = next(net.parameters()).dtype

    # collect decisive corrections (A = mcts_top, B = policy move)
    corrs = []
    for f in sorted(glob.glob(a.glob)):
        try: d = json.load(open(f))
        except Exception: continue
        for c in d['corrections']:
            if c['mcts_top_idx'] == c['pol_idx']:
                continue
            if (c['mcts_top_share'] - c['pol_share']) < a.min_margin:
                continue
            corrs.append(c)
        if len(corrs) >= a.n * 3:
            break
    if not corrs:
        print("no decisive corrections found"); return
    rng = np.random.default_rng(0)
    idx = rng.choice(len(corrs), size=min(a.n, len(corrs)), replace=False)
    sample = [corrs[i] for i in idx]
    print(f"Testing {len(sample)} decisive corrections (margin>={a.min_margin}), "
          f"R={a.R} common-RNG rollouts, horizon {a.horizon}, pillar3f greedy", flush=True)

    # build jobs: for each correction, A-move and B-move over the SAME R seeds
    jobs, meta = [], []
    for k, c in enumerate(sample):
        anchor = {'board': c['board'], 'next_balls': c['next_balls'],
                  'score': 0, 'turn': int(c['turn'])}
        A = _decode(int(c['mcts_top_idx'])); B = _decode(int(c['pol_idx']))
        for tag, mv in (('A', A), ('B', B)):
            for s in range(a.R):
                jobs.append((anchor, mv, a.seed_base + s))
                meta.append((k, tag))
    res = batched_rollout(net, dev, dtype, jobs, a.horizon, batch=a.batch)

    # aggregate per correction
    per = {}
    for (k, tag), r in zip(meta, res):
        per.setdefault(k, {'A': [], 'B': []})[tag].append(r)
    A_better_turns = A_better_cat = n_valid = 0
    dturns, dcat = [], []
    for k, dd in per.items():
        At = np.array([r['turns'] for r in dd['A']]); Ad = np.array([r['died'] for r in dd['A']])
        Bt = np.array([r['turns'] for r in dd['B']]); Bd = np.array([r['died'] for r in dd['B']])
        n_valid += 1
        dt = At.mean() - Bt.mean()                  # >0 => A survives longer
        dc = 100*(Ad.mean() - Bd.mean())            # <0 => A safer (lower catastrophe)
        dturns.append(dt); dcat.append(dc)
        A_better_turns += dt > 0
        A_better_cat += dc < 0
    dturns = np.array(dturns); dcat = np.array(dcat)
    print(f"\n=== VERDICT over {n_valid} decisive corrections ===", flush=True)
    print(f"A (MCTS move) survives LONGER than B (pillar3f move): "
          f"{A_better_turns}/{n_valid} = {100*A_better_turns/n_valid:.0f}%", flush=True)
    print(f"A has LOWER catastrophe than B:                       "
          f"{A_better_cat}/{n_valid} = {100*A_better_cat/n_valid:.0f}%", flush=True)
    print(f"mean turns delta (A-B): mean={dturns.mean():+.0f} median={np.median(dturns):+.0f} "
          f"[P25 {np.percentile(dturns,25):+.0f}, P75 {np.percentile(dturns,75):+.0f}]", flush=True)
    print(f"mean catastrophe delta (A-B) pp: mean={dcat.mean():+.1f} median={np.median(dcat):+.1f}", flush=True)
    print("\n--- READ ---", flush=True)
    print("A beats B on ~MOST corrections (>>50%, positive turns / negative cat) => corrections")
    print("  are GENUINE for pillar3f => the bug is DISTILLATION/merge collateral. (Gemini/ChatGPT.)")
    print("A ~= B (near 50%, deltas ~0) => MCTS-top is NOT actually better => FV teacher too weak")
    print("  for the strong pillar3f => need a stronger value head. No paradox.")


if __name__ == '__main__':
    main()
