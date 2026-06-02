"""Held-out fork-ranking eval: does the policy rank the confirmed SAFE move
above the risky move it actually played?

This is the metric we trust for the floor work — NOT validation CE. pillar3c
regressed while its val loss looked fine (project_pillar3c_failure.md), because
the rare high-value forks were invisible in an averaged loss. Here we score the
model directly on the confirmed crisis forks, split BY SEED so the held-out
games test generalization (anchors from one game are correlated; an anchor-level
split would leak).

Per fork we have: winner = confirmed lower-catastrophe move, pol = the move the
policy played (higher catastrophe), and clean losers = clearly-worse moves.
We report, per split:
  win_rank   median rank of the winner among legal moves (1 = would be played)
  top1%      winner is the argmax (policy would now play the safe move)
  top5%      winner in policy's top 5
  margin     mean (pol_logit - winner_logit); >0 = policy prefers the risky
             move (the gap we must close); <0 after a good fix
  flip%      winner_logit > pol_logit (policy now prefers safe over its old pick)
  conc       winner ranked above clean losers (pairwise concordance / AUC)

Run on the baseline (pillar3b) to size the opportunity; re-run on a trained
checkpoint to measure whether the signal was learned and generalized.

    PYTHONPATH=. python scripts/eval_fork_ranking.py \\
        --model alphatrain/data/pillar3b_epoch_20.pt --fp16
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.evaluate import load_model
from alphatrain.counterfactual import build_crisis_corpus
from alphatrain.mcts import _build_obs_for_game
from scripts.batched_rollout import restore


def _next_balls_from_row(npos, ncol, nn):
    return [((int(npos[i][0]), int(npos[i][1])), int(ncol[i])) for i in range(int(nn))]


def _legal_flat(game):
    out = []
    for (sr, sc), (tr, tc) in game.get_legal_moves():
        out.append((sr * 9 + sc) * 81 + (tr * 9 + tc))
    return out


@torch.no_grad()
def eval_corpus(net, device, dtype, corpus):
    """Return per-anchor dict of metrics + the seed, for later split aggregation."""
    boards = corpus['boards'].cpu().numpy()
    npos = corpus['next_pos'].cpu().numpy()
    ncol = corpus['next_col'].cpu().numpy()
    nn = corpus['n_next'].cpu().numpy()
    win = corpus['winner_idx'].cpu().numpy()
    pol = corpus['top1_idx'].cpu().numpy()
    losers = corpus['loser_idx'].cpu().numpy()
    lmask = corpus['loser_mask'].cpu().numpy()
    seeds = corpus['seed'].cpu().numpy()
    weights = corpus['weight'].cpu().numpy()
    N = len(boards)

    rows = []
    for i in range(N):
        anchor = {'board': boards[i].tolist(),
                  'next_balls': _next_balls_from_row(npos[i], ncol[i], nn[i]),
                  'score': 0, 'turn': 0}
        g = restore(anchor, 0)
        obs = torch.from_numpy(_build_obs_for_game(g)).unsqueeze(0).to(device, dtype)
        logits = net(obs)[0].float().cpu().numpy()
        legal = _legal_flat(g)
        legal_logits = np.array([logits[m] for m in legal])
        wl = logits[win[i]]
        pl = logits[pol[i]]
        win_rank = int((legal_logits > wl).sum()) + 1     # 1 = argmax
        clean = [losers[i][k] for k in range(losers.shape[1]) if lmask[i][k]]
        conc = (np.mean([1.0 if wl > logits[c] else 0.0 for c in clean])
                if clean else np.nan)
        rows.append({'seed': int(seeds[i]), 'weight': float(weights[i]),
                     'win_rank': win_rank, 'top1': win_rank == 1,
                     'top5': win_rank <= 5, 'margin': float(pl - wl),
                     'flip': wl > pl, 'conc': conc})
    return rows


def _agg(rows, label):
    if not rows:
        print(f"  {label:>9}: (empty)")
        return
    wr = np.array([r['win_rank'] for r in rows])
    mg = np.array([r['margin'] for r in rows])
    fl = np.array([r['flip'] for r in rows], float)
    t1 = np.array([r['top1'] for r in rows], float)
    t5 = np.array([r['top5'] for r in rows], float)
    cc = np.array([r['conc'] for r in rows], float)
    cc = cc[~np.isnan(cc)]
    print(f"  {label:>9}: n={len(rows):>4}  win_rank med={np.median(wr):>4.0f} "
          f"mean={wr.mean():>5.1f}  top1={100*t1.mean():>4.0f}% "
          f"top5={100*t5.mean():>4.0f}%  margin={mg.mean():>5.2f}  "
          f"flip={100*fl.mean():>4.0f}%  conc={100*np.nanmean(cc) if cc.size else float('nan'):>4.0f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--corpus', default=None,
                   help='Pre-built crisis corpus .pt (build_crisis_corpus_file). '
                        'If set, skips the mine/death build (Colab-friendly).')
    p.add_argument('--mine-glob', default='logs/mine_*.json')
    p.add_argument('--death-dir', default='alphatrain/data/death_games')
    p.add_argument('--clean-loser-margin', type=float, default=10.0)
    p.add_argument('--holdout-frac', type=float, default=0.25)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--fp16', action='store_true')
    a = p.parse_args()

    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    net, _ = load_model(a.model, dev, fp16=a.fp16)
    dtype = next(net.parameters()).dtype

    if a.corpus:
        corpus = torch.load(a.corpus, map_location='cpu', weights_only=False)
    else:
        corpus = build_crisis_corpus(a.mine_glob, a.death_dir, device='cpu',
                                     clean_loser_margin=a.clean_loser_margin)
    st = corpus['_stats']
    print(f"corpus: {st['n_anchors']} confirmed forks from {st['n_seeds']} "
          f"seeds / {st['n_files']} games; {st['n_clean_pairs']} clean-loser "
          f"pairs; dropped {st['n_unconfirmed']} unconfirmed, "
          f"{st['n_degenerate']} degenerate\n", flush=True)

    # by-seed split
    uniq = sorted(set(int(s) for s in corpus['seed'].tolist()))
    rng = np.random.default_rng(a.split_seed)
    perm = rng.permutation(len(uniq))
    n_hold = max(1, int(round(a.holdout_frac * len(uniq))))
    hold = set(uniq[i] for i in perm[:n_hold])

    rows = eval_corpus(net, dev, dtype, corpus)
    train_rows = [r for r in rows if r['seed'] not in hold]
    hold_rows = [r for r in rows if r['seed'] in hold]

    print(f"model: {os.path.basename(a.model)}  fp{'16' if a.fp16 else '32'}  "
          f"(holdout {n_hold}/{len(uniq)} seeds)")
    print("  win_rank: where policy ranks the confirmed SAFE move "
          "(1=would play it; high=buried)")
    _agg(rows, 'ALL')
    _agg(train_rows, 'train')
    _agg(hold_rows, 'heldout')


if __name__ == '__main__':
    main()
