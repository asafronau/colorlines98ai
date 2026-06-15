"""Is eval_parallel's MCTS actually using its sim budget, and does it differ from
selfplay's MCTS at equal sims? Head-to-head on FIXED boards — no cap, no seed-set,
no harness confound.

Uses crisis-corpus boards (states where 4800-sim MCTS chose a move != the policy, so
there IS a better move for search to find). For each board, on the SAME MCTS object:
  pol         = policy argmax-legal (0 sims)
  eval@S      = search(game)                          # eval_parallel path: temp 0, return_policy F
  sp@S        = search(game, temp 0, dir 0.3/0.25, return_policy T)   # selfplay path
  sp_nodir@S  = search(game, temp 0, return_policy T)  # selfplay path, Dirichlet off
Reports root visit-sum (does it scale with S?), and match rates between configs and
against pol / the stored 4800-move. Conclusions:
  - root visits ~= S  => sims are consumed (not frozen).
  - eval@S == sp_nodir@S  => the two harnesses run identical search (no eval bug).
  - eval@400 != eval@100 and drifts toward the 4800-move => sims change the decision.

    PYTHONPATH=. python scripts/diag_mcts_sims.py --n 40 --device mps
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.evaluate import load_model
from alphatrain.mcts import MCTS, _get_legal_priors_flat, _build_obs_for_game


def set_board(corpus, i, device):
    g = ColorLinesGame()
    board = corpus['boards'][i].numpy().astype(np.int8)
    npos = corpus['next_pos'][i].numpy()
    ncol = corpus['next_col'][i].numpy()
    nn = int(corpus['n_next'][i])
    nb = [((int(npos[j, 0]), int(npos[j, 1])), int(ncol[j])) for j in range(nn)]
    g.reset(board=board, next_balls=nb)
    g.score, g.turns = 0, 5000
    return g


def root_visits(mcts):
    return sum(c.visit_count for c in mcts._last_root.children.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default='crisis/corrections_corpus_mm05.pt')
    p.add_argument('--model', default='alphatrain/data/pillar3f.pt')
    p.add_argument('--value-head', default='alphatrain/data/value_head_pillar3f.pt')
    p.add_argument('--n', type=int, default=40)
    p.add_argument('--sims', type=int, nargs='+', default=[100, 400])
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)
    corpus = torch.load(a.corpus, map_location='cpu', weights_only=False)
    net, ms = load_model(a.model, dev, fp16=True, jit_trace=False)

    def mk(sims):
        return MCTS(net, dev, max_score=ms, num_simulations=sims, batch_size=8,
                    top_k=30, c_puct=2.5, value_head_path=a.value_head, q_weight=2.0)

    idx = list(range(min(a.n, corpus['boards'].shape[0])))
    mcts4800 = corpus['tgt_idx'][:, 0]

    for S in a.sims:
        m = mk(S)
        vis_sum = []
        ev, spd, sp, pol = [], [], [], []
        for i in idx:
            g = set_board(corpus, i, dev)
            pol_np, _ = m._nn_evaluate_single(g)  # priors dict (legal)
            pol_mv = max(pol_np.items(), key=lambda x: x[1])[0]
            pol.append(pol_mv)
            e = m.search(set_board(corpus, i, dev)); ev.append(_flat(e))
            vis_sum.append(root_visits(m))
            d = m.search(set_board(corpus, i, dev), temperature=0.0,
                         return_policy=True)
            spd.append(_flat(d[0]))
            s = m.search(set_board(corpus, i, dev), temperature=0.0,
                         dirichlet_alpha=0.3, dirichlet_weight=0.25,
                         return_policy=True)
            sp.append(_flat(s[0]))
        ev = np.array(ev); spd = np.array(spd); sp = np.array(sp)
        pol = np.array(pol); m4 = mcts4800[idx].numpy()
        vis = np.array(vis_sum)
        print(f"\n=== sims={S} (n={len(idx)}) ===")
        print(f"  root visit-sum: mean {vis.mean():.0f} (min {vis.min()} "
              f"max {vis.max()})   [should be ~{S} if sims are consumed]")
        print(f"  eval == selfplay(no-dir)  : {(ev==spd).mean()*100:.0f}%  "
              f"[100% => identical search, no harness bug]")
        print(f"  eval == selfplay(+dir)    : {(ev==sp).mean()*100:.0f}%  "
              f"[<100% = Dirichlet noise only]")
        print(f"  eval == policy            : {(ev==pol).mean()*100:.0f}%  "
              f"[100% => MCTS never overrides policy]")
        print(f"  eval == 4800-move         : {(ev==m4).mean()*100:.0f}%  "
              f"[rises with sims => search converging to deep answer]")


def _flat(action):
    if action is None:
        return -1
    (sr, sc), (tr, tc) = action
    return (sr * 9 + sc) * 81 + (tr * 9 + tc)


if __name__ == '__main__':
    main()
