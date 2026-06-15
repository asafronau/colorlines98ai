"""Experiment #1 (ChatGPT/Gemini review): is the VISIT target prior-contaminated?

On a sample of V14 states, compare three candidate-rankings for the top move:
  policy-top1 = pillar3f greedy argmax (the prior)
  visit-top1  = the stored most-visited move (the distillation target's argmax)
  Q-top1      = argmax over candidates of afterstate feature-value (the IMPROVEMENT signal)

Prior-domination predicts: visit-top1 ≈ policy-top1 (visits just smooth the prior), while
Q-top1 disagrees with the policy MUCH more (the search's value insight). If
disagree(Q,policy) >> disagree(visit,policy), the visit target carries little improvement
signal and a Q/advantage target is the fix — validating the offline relabel before any
training spend.

Q(a) proxy = feature-value of the afterstate (apply move a via the engine, evaluate with the
SAME linear leaf evaluator the MCTS used). 1-ply, but the value is saturated so root Q ≈ this.
Candidate set = the corpus's stored top-K visit moves (pol_indices) — what the search ranked.

    PYTHONPATH=. python scripts/diag_q_vs_visit.py --n 1500 --device mps
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from game.board import ColorLinesGame, _clear_lines_at
from alphatrain.evaluate import load_model
from alphatrain.mcts import MCTS, _evaluate_features_linear

FV = 'alphatrain/data/feature_value_weights_2y_nb.npz'


def game_from_row(boards, npos, ncol, nn, i):
    g = ColorLinesGame(seed=12345)  # fixed seed: displacement during spawn is deterministic
    nb = [((int(npos[i, j, 0]), int(npos[i, j, 1])), int(ncol[i, j]))
          for j in range(int(nn[i]))]
    g.reset(board=boards[i].astype(np.int8), next_balls=nb)
    return g


def _feat(mcts, board, next_balls):
    gg = ColorLinesGame()
    gg.reset(board=board, next_balls=next_balls)
    n = mcts._fill_next_ball_buffers(gg)
    return float(_evaluate_features_linear(
        board, mcts._nb_r, mcts._nb_c, mcts._nb_col, n,
        mcts.feature_coefs, mcts.feature_means, mcts.feature_stds, mcts.feature_bias))


def afterstate_q_postspawn(mcts, g, flat):
    """feature-value after the FULL move (move+clear+spawn+regen), one fixed-seed determinization."""
    src, tgt = flat // 81, flat % 81
    gg = ColorLinesGame(seed=12345)
    gg.reset(board=g.board.copy(),
             next_balls=[(tuple(p), int(c)) for p, c in g.next_balls])
    r = gg.move((src // 9, src % 9), (tgt // 9, tgt % 9))
    if not r['valid']:
        return -1e9
    return _feat(mcts, gg.board, [(tuple(p), int(c)) for p, c in gg.next_balls])


def afterstate_q_clean(mcts, g, flat):
    """feature-value after move+clear ONLY (no spawn) — deterministic, evaluated with the
    ORIGINAL pending next_balls. Removes spawn/regen RNG from the move-comparison."""
    src, tgt = flat // 81, flat % 81
    b = g.board.copy()
    sr, sc, tr, tc = src // 9, src % 9, tgt // 9, tgt % 9
    if b[sr, sc] == 0 or b[tr, tc] != 0:
        return -1e9
    b[tr, tc] = b[sr, sc]; b[sr, sc] = 0
    _clear_lines_at(b, tr, tc)
    return _feat(mcts, b, [(tuple(p), int(c)) for p, c in g.next_balls])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', default='alphatrain/data/v14_pillar3f.pt')
    p.add_argument('--model', default='alphatrain/data/pillar3f.pt')
    p.add_argument('--n', type=int, default=1500)
    p.add_argument('--device', default='mps')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    dev = torch.device(a.device)

    t = torch.load(a.tensor, map_location='cpu', weights_only=False)
    N = t['boards'].shape[0]
    rng = np.random.RandomState(a.seed)
    idx = rng.choice(N, size=min(a.n, N), replace=False)
    boards = t['boards'][idx].numpy()
    npos = t['next_pos'][idx].numpy().astype(np.int64)
    ncol = t['next_col'][idx].numpy().astype(np.int64)
    nn = t['n_next'][idx].numpy()
    pol_idx = t['pol_indices'][idx].numpy()    # [n,5] flat, visit-ranked
    pol_nnz = t['pol_nnz'][idx].numpy()

    net, _ = load_model(a.model, dev, fp16=False)
    mcts = MCTS(net, dev, num_simulations=1, top_k=30, feature_weights_path=FV)

    d_visit_pol = 0
    # for each proxy: disagreement with prior, and with the SEARCH decision (visit-top1)
    dp = {'post': [0, 0], 'clean': [0, 0]}   # [vs policy, vs visit]
    n_used = 0
    for k in range(len(idx)):
        g = game_from_row(boards, npos, ncol, nn, k)
        priors, _ = mcts._nn_evaluate_single(g)
        if not priors:
            continue
        policy_top1 = max(priors, key=priors.get)
        cands = [int(pol_idx[k, j]) for j in range(int(pol_nnz[k])) if pol_idx[k, j] >= 0]
        if not cands:
            continue
        visit_top1 = cands[0]                    # corpus stores visit-descending = MCTS argmax
        n_used += 1
        d_visit_pol += (visit_top1 != policy_top1)
        for name, fn in (('post', afterstate_q_postspawn), ('clean', afterstate_q_clean)):
            qs = {c: fn(mcts, g, c) for c in cands}
            qtop = max(qs, key=qs.get)
            dp[name][0] += (qtop != policy_top1)
            dp[name][1] += (qtop != visit_top1)

    print(f"\nV14 sample: {n_used} states (candidates = stored top-K visit moves)\n")
    print(f"  disagree(visit-top1, policy-top1) = {100*d_visit_pol/n_used:.1f}%   "
          f"[visits vs prior — low ⇒ visit target just tracks the prior]\n")
    print(f"  {'proxy':<22}{'vs policy(prior)':>18}{'vs visit(SEARCH)':>18}")
    for name in ('post', 'clean'):
        print(f"  afterstate-Q [{name:<5}]     {100*dp[name][0]/n_used:>16.1f}%"
              f"{100*dp[name][1]/n_used:>17.1f}%")
    print(f"\n  Read: 'vs SEARCH' LOW ⇒ the cheap proxy reproduces MCTS@400's decisions ⇒ "
          f"offline relabel viable.\n        'vs SEARCH' HIGH ⇒ proxy ≠ the search ⇒ need REAL "
          f"root Q (regenerate).")


if __name__ == '__main__':
    main()
