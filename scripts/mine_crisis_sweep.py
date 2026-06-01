"""Crisis-fork miner (integrated) — adaptive band + batched engine + screen/confirm.

Per recorded death game, over rewind depths [--lo..--hi]:
  A. POLICY-CURVE scan (cheap): roll out the policy's recorded move at each depth
     (R_curve seeds) → catastrophe-vs-depth → keep the RECOVERABLE band
     (catastrophe in [--band-lo, --band-hi]%). This adaptively finds *this* game's
     crisis-onset instead of a fixed grid (40006's was shallow; early deaths mid).
  B. SCREEN: at each band depth, eval candidates (policy top-K ∪ feature-value
     net-widener) at R_screen seeds; flag a fork if some candidate beats the
     policy move by >= --flag-gap pp catastrophe.
  C. CONFIRM: re-eval each flagged (policy, best) pair on FRESH seeds (R_confirm),
     paired bootstrap; keep only forks whose CI excludes 0 (winner's-curse guard).

catastrophe = DIED within --horizon turns of the rewind point (anchor-relative).
All rollouts go through the validated single-process batched engine (fp16).
Every (board, next_balls, [(move, catastrophe%)]) is a teacher label.

Usage:
    PYTHONPATH=. python scripts/mine_crisis_sweep.py \\
        --game alphatrain/data/death_games/death_21585.json --fp16
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from game.board import ColorLinesGame, _clear_lines_at
from alphatrain.evaluate import load_model
from alphatrain.mcts import (_build_obs_for_game, _get_legal_priors_flat,
                             _evaluate_features_linear)
from scripts.batched_rollout import batched_rollout, restore, _decode


def _next_arrays(next_balls):
    nr = np.zeros(3, dtype=np.intp); nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp); nn = min(len(next_balls), 3)
    for i in range(nn):
        p, c = next_balls[i]
        nr[i], nc[i], ncol[i] = int(p[0]), int(p[1]), int(c)
    return nr, nc, ncol, nn


def gen_candidates(net, dev, dtype, anchor, recorded_move, pol_k, fv_k, fvw):
    """Candidate moves = recorded policy move ∪ policy top-K ∪ feature-value
    top-K (the net-widener that surfaces survival-good moves the policy under-
    ranks). Returns (candidate_moves, policy_index)."""
    g = restore(anchor, 0)
    obs = torch.from_numpy(_build_obs_for_game(g)).unsqueeze(0).to(dev, dtype)
    with torch.no_grad():
        logits = net(obs)[0].float().cpu().numpy()
    pri = _get_legal_priors_flat(g.board, logits, 64)
    cand = [_decode(m) for m, _ in sorted(pri.items(), key=lambda x: -x[1])[:pol_k]]
    if fv_k > 0 and fvw is not None:
        nr, nc, ncol, nn = _next_arrays(anchor['next_balls'])
        b0 = np.array(anchor['board'], dtype=np.int8)
        scored = []
        for (sr, sc), (tr, tc) in g.get_legal_moves():
            aft = b0.copy()
            col = aft[sr, sc]; aft[sr, sc] = 0; aft[tr, tc] = col
            _clear_lines_at(aft, int(tr), int(tc))
            v = _evaluate_features_linear(aft, nr, nc, ncol, nn, *fvw)
            scored.append((float(v), ((sr, sc), (tr, tc))))
        scored.sort(key=lambda x: -x[0])
        for _, mv in scored[:fv_k]:
            if mv not in cand:
                cand.append(mv)
    if recorded_move not in cand:
        cand.append(recorded_move)
    return cand, cand.index(recorded_move)


def catrate(died_list):
    return float(100.0 * np.mean(died_list)) if died_list else float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game', required=True)
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--fv-weights',
                   default='alphatrain/data/feature_value_weights_2y_nb.npz')
    p.add_argument('--lo', type=int, default=15)
    p.add_argument('--hi', type=int, default=45)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--r-curve', type=int, default=100)
    p.add_argument('--r-screen', type=int, default=100)
    p.add_argument('--r-confirm', type=int, default=500)
    p.add_argument('--pol-k', type=int, default=10)
    p.add_argument('--fv-k', type=int, default=12)
    p.add_argument('--band-lo', type=float, default=15.0)
    p.add_argument('--band-hi', type=float, default=85.0)
    p.add_argument('--flag-gap', type=float, default=10.0)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    t0 = time.time()

    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    net, _ = load_model(a.model, dev, fp16=a.fp16)
    dtype = next(net.parameters()).dtype
    fvw = None
    if a.fv_k > 0:
        w = np.load(a.fv_weights)
        assert w['coefs'].shape[0] == 27, "need 27-feature fv weights (…_2y_nb.npz)"
        fvw = (w['coefs'].astype(np.float32), w['means'].astype(np.float32),
               w['stds'].astype(np.float32), float(w['bias']))

    d = json.load(open(a.game))
    frames = d['frames']
    death_idx = len(frames) - 1
    out = a.out or f"logs/mine_{os.path.basename(a.game).split('.')[0]}.json"

    # anchors per depth + recorded policy move
    depth_anchor = {}
    for depth in range(a.lo, a.hi + 1):
        idx = death_idx - depth
        if idx < 0:
            continue
        fr = frames[idx]
        depth_anchor[depth] = (
            {'board': fr['board'], 'next_balls': fr['next_balls'],
             'score': fr.get('score_before', fr['score']), 'turn': fr['turn']},
            tuple(map(tuple, fr['chosen_move'])), fr['turn'], fr.get('empties'))
    depths = sorted(depth_anchor)
    print(f"seed={d['seed']} death {d['final_score']}@{d['final_turns']}; "
          f"depths {a.lo}-{a.hi}; horizon {a.horizon}; "
          f"R curve/screen/confirm={a.r_curve}/{a.r_screen}/{a.r_confirm}; "
          f"band [{a.band_lo},{a.band_hi}]%; fp{'16' if a.fp16 else '32'} -> {out}",
          flush=True)

    def run(jobs, tag):
        res = batched_rollout(net, dev, dtype, jobs, a.horizon, batch=a.batch)
        print(f"  [{tag}] {len(jobs)} rollouts, {time.time()-t0:.0f}s", flush=True)
        return res

    # ── Phase A: policy-curve → recoverable band ──
    jobsA, idxA = [], {}
    for depth in depths:
        anchor, pmove, _, _ = depth_anchor[depth]
        idxA[depth] = (len(jobsA), len(jobsA) + a.r_curve)
        jobsA += [(anchor, pmove, s) for s in range(a.r_curve)]
    resA = run(jobsA, 'policy-curve')
    curve = {}
    for depth in depths:
        lo, hi = idxA[depth]
        curve[depth] = catrate([resA[i]['died'] for i in range(lo, hi)])
    band = [depth for depth in depths if a.band_lo <= curve[depth] <= a.band_hi]
    print("  policy catastrophe vs depth: "
          + " ".join(f"d{depth}:{curve[depth]:.0f}%" for depth in depths), flush=True)
    print(f"  recoverable band ({a.band_lo}-{a.band_hi}%): "
          f"{band or 'EMPTY (all sealed/safe)'}", flush=True)

    # ── Phase B: candidate screen on band depths ──
    rows, flagged = [], []
    jobsB, idxB, cand_of = [], {}, {}
    for depth in band:
        anchor, pmove, turn, empt = depth_anchor[depth]
        cand, pol_i = gen_candidates(net, dev, dtype, anchor, pmove,
                                     a.pol_k, a.fv_k, fvw)
        cand_of[depth] = (cand, pol_i)
        for ci, mv in enumerate(cand):
            idxB[(depth, ci)] = (len(jobsB), len(jobsB) + a.r_screen)
            jobsB += [(anchor, mv, s) for s in range(a.r_screen)]
    resB = run(jobsB, 'screen') if jobsB else []
    for depth in band:
        anchor, pmove, turn, empt = depth_anchor[depth]
        cand, pol_i = cand_of[depth]
        cat = []
        for ci in range(len(cand)):
            lo, hi = idxB[(depth, ci)]
            cat.append(catrate([resB[i]['died'] for i in range(lo, hi)]))
        pol_cat = cat[pol_i]
        best_ci = int(np.argmin(cat))
        gap = pol_cat - cat[best_ci]
        flag = best_ci != pol_i and gap >= a.flag_gap
        rows.append({'depth': depth, 'turn': turn, 'empties': empt,
                     'board': anchor['board'], 'next_balls': anchor['next_balls'],
                     'pol_cat': pol_cat, 'best_cat': cat[best_ci],
                     'best_move': list(map(list, cand[best_ci])), 'gap': gap,
                     'flag': flag,
                     'cand_rates': [[list(map(list, cand[ci])), cat[ci], a.r_screen]
                                    for ci in range(len(cand))]})
        if flag:
            flagged.append((depth, cand[pol_i], cand[best_ci]))

    # ── Phase C: confirm flagged forks on FRESH seeds ──
    jobsC, idxC = [], {}
    for k, (depth, pmove, bmove) in enumerate(flagged):
        anchor = depth_anchor[depth][0]
        for tagmv in (pmove, bmove):
            idxC[(k, tagmv)] = (len(jobsC), len(jobsC) + a.r_confirm)
            jobsC += [(anchor, tagmv, s)
                      for s in range(a.r_screen, a.r_screen + a.r_confirm)]
    resC = run(jobsC, 'confirm') if jobsC else []
    rng = np.random.default_rng(0)
    confirms = {}
    for k, (depth, pmove, bmove) in enumerate(flagged):
        pl, ph = idxC[(k, pmove)]; bl, bh = idxC[(k, bmove)]
        polc = np.array([resC[i]['died'] for i in range(pl, ph)], float)
        bestc = np.array([resC[i]['died'] for i in range(bl, bh)], float)
        n = len(polc)
        boot = np.array([100*(polc[ix].mean()-bestc[ix].mean())
                         for ix in (rng.integers(0, n, n) for _ in range(2000))])
        lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
        confirms[depth] = {'gap': float(100*(polc.mean()-bestc.mean())),
                           'pol_cat': float(100*polc.mean()),
                           'best_cat': float(100*bestc.mean()),
                           'lo': float(lo_ci), 'hi': float(hi_ci),
                           'real': bool(lo_ci > 0)}

    # ── report + save ──
    print(f"\n{'depth':>5} {'turn':>5} {'empt':>4} {'pol%':>6} {'best%':>6} "
          f"{'gap':>5}  {'bestMove':>10} {'verdict':>22}", flush=True)
    print('-' * 78, flush=True)
    n_real = 0
    for r in rows:
        v = 'neutral'
        if r['flag']:
            c = confirms.get(r['depth'])
            if c and c['real']:
                v = f"REAL Δ{c['gap']:.0f}[{c['lo']:.0f},{c['hi']:.0f}]"; n_real += 1
            else:
                v = f"curse Δ{c['gap']:.0f}[{c['lo']:.0f},{c['hi']:.0f}]" if c else 'flag'
        print(f"{r['depth']:>5} {r['turn']:>5} {str(r['empties']):>4} "
              f"{r['pol_cat']:>6.1f} {r['best_cat']:>6.1f} {r['gap']:>5.1f}  "
              f"{str(tuple(map(tuple, r['best_move']))):>10} {v:>22}", flush=True)

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({'meta': {'game': a.game, 'seed': d['seed'],
                        'death': [d['final_score'], d['final_turns']],
                        'horizon': a.horizon, 'band': band, 'curve': curve,
                        'r_screen': a.r_screen, 'r_confirm': a.r_confirm,
                        'fp16': a.fp16},
               'rows': rows, 'confirms': {str(k): v for k, v in confirms.items()}},
              open(out, 'w'), default=float)
    print(f"\n{n_real} REAL forks; band {len(band)} depths; "
          f"{sum(len(r['cand_rates']) for r in rows)} per-move labels; "
          f"{time.time()-t0:.0f}s. Wrote {out}", flush=True)


if __name__ == '__main__':
    main()
