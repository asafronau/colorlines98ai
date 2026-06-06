"""Golden test for MCTS: prove a refactor is BIT-IDENTICAL, not just argmax-stable.

The lesson from the batch_size/top_k checks: argmax can agree while the soft visit
distribution (our training target) still shifts. A change is "provably safe" ONLY if
the visit distribution is bit-for-bit identical. This freezes the distributions from
the current code (capture) and re-checks them after a refactor (check), asserting
exact equality. Dirichlet noise is seeded so single-seed search is reproducible.

    PYTHONPATH=. python scripts/golden_search_test.py capture   # before refactor
    PYTHONPATH=. python scripts/golden_search_test.py check     # after refactor
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

MODEL = 'alphatrain/data/pillar3b_epoch_20.pt'
FV = 'alphatrain/data/feature_value_weights_2y_nb.npz'
GOLDEN = 'crisis/golden_search.npz'
NP_SEED = 1234  # seed the Dirichlet noise so single-seed search is reproducible


def _probes(death_glob, depths):
    """Deterministic probe list (fixed glob order) so capture/check align by index."""
    out = []
    for f in sorted(glob.glob(death_glob))[:4]:
        g = json.load(open(f)); nf = len(g['frames'])
        for d in depths:
            i = (nf - 1) - d
            if i < 0:
                continue
            fr = g['frames'][i]
            if fr.get('chosen_move') is None:
                continue
            out.append((g['seed'], d, fr))
    return out


def _run(mcts, fr):
    from game.board import ColorLinesGame
    game = ColorLinesGame()
    game.reset(board=np.array(fr['board'], dtype=np.int8),
               next_balls=[(tuple(p), int(c)) for p, c in fr['next_balls']])
    game.score, game.turns = 0, int(fr['turn'])
    np.random.seed(NP_SEED)
    _, pt = mcts.search(game, temperature=0.0, dirichlet_alpha=0.3,
                        dirichlet_weight=0.25, return_policy=True)
    return np.asarray(pt, dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['capture', 'check'])
    p.add_argument('--model', default=MODEL)
    p.add_argument('--death-glob', default='crisis/death_games/death_*.json')
    p.add_argument('--top-k', type=int, default=300)
    p.add_argument('--sims', type=int, default=4800)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='mps')
    a = p.parse_args()

    from alphatrain.evaluate import load_model
    from alphatrain.mcts import MCTS
    dev = torch.device(a.device)
    net, _ = load_model(a.model, dev, fp16=(dev.type != 'cpu'))
    mcts = MCTS(net=net, device=dev, num_simulations=a.sims, c_puct=2.5,
                top_k=a.top_k, batch_size=a.batch_size,
                feature_weights_path=FV, q_weight=2.0)
    probes = _probes(a.death_glob, depths=(30, 50, 70))
    print(f"{a.mode}: {len(probes)} probes, top_k={a.top_k} "
          f"bs={a.batch_size} sims={a.sims} device={a.device}", flush=True)

    if a.mode == 'capture':
        dists = np.stack([_run(mcts, fr) for (_, _, fr) in probes])
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        np.savez(GOLDEN, dists=dists)
        print(f"  saved {dists.shape} -> {GOLDEN}", flush=True)
        return

    gold = np.load(GOLDEN)['dists']
    bad = 0
    for k, (sd, d, fr) in enumerate(probes):
        cur = _run(mcts, fr)
        identical = np.array_equal(cur, gold[k])
        diff = np.abs(cur - gold[k])
        if not identical:
            bad += 1
        print(f"  [{k+1}/{len(probes)}] {sd}:D-{d}  "
              + ('IDENTICAL' if identical
                 else f'DIFFERS (TV={0.5*diff.sum():.4f}, maxabs={diff.max():.2e})'),
              flush=True)
    print(f"\n=== {'ALL BIT-IDENTICAL (provably safe)' if bad==0 else f'{bad} DIFFER — NOT safe'} ===",
          flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
