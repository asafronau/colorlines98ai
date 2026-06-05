"""Generate the MCTS-corrections teacher corpus (the floor pipeline, v2).

Rewind policy death games to the crisis band (D-15..D-85); at each band state run
WIDENED deep MCTS (top_k = all legal + Dirichlet noise, MULTI-SEED for RNG
robustness). Keep ONLY corrections (MCTS's top move != the policy's recorded
move) -- we do not teach what the policy already plays (the base-corpus
distillation anchors that). The soft MCTS visit distribution IS the target. MCTS
is phantom-free and spawn-robust (validated 2026-06-03), replacing the
greedy-mined forks (~40% of which were phantoms).

Per-game output (crisis/corrections/corr_<seed>.json) => resumable + parallelizable
across machines (each does a disjoint death-game subset). Everything under crisis/.

    PYTHONPATH=. python scripts/gen_mcts_corrections.py \\
        --death-glob 'crisis/death_games/death_*.json' --sims 4800 --mcts-seeds 3
"""
import os, sys, glob, json, time, argparse, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.evaluate import load_model
from alphatrain.mcts import MCTS
from scripts.batched_rollout import restore

FV = 'alphatrain/data/feature_value_weights_2y_nb.npz'
MODEL = 'alphatrain/data/pillar3b_epoch_20.pt'


def _flat(m):
    return (m[0][0] * 9 + m[0][1]) * 81 + (m[1][0] * 9 + m[1][1])


def process_game(g, seed, net, dev, a):
    nf = len(g['frames'])
    out, n_states, max_depth = [], 0, 0
    for i, fr in enumerate(g['frames']):
        depth = (nf - 1) - i
        max_depth = max(max_depth, depth)
        if not (a.lo <= depth <= a.hi):
            continue
        pol = fr.get('chosen_move')
        if pol is None:
            continue
        n_states += 1
        anchor = {'board': fr['board'],
                  'next_balls': [[list(pp), int(cc)] for pp, cc in fr['next_balls']],
                  'score': 0, 'turn': fr['turn']}
        # multi-seed: average visit dists over K determinized spawn streams. Vary the
        # MCTS state-hash seed by perturbing game.turns (hash input only, not the board).
        visit_sum = np.zeros(6561, dtype=np.float64)
        for s in range(a.mcts_seeds):
            game = restore(anchor, 0)
            game.turns = int(fr['turn']) + s * 1_000_003
            nl = len(game.get_legal_moves())
            mcts = MCTS(net=net, device=dev, num_simulations=a.sims, c_puct=2.5,
                        top_k=max(300, nl), batch_size=16,
                        feature_weights_path=FV, q_weight=2.0)
            _, pt = mcts.search(game, temperature=0.0, dirichlet_alpha=0.3,
                                dirichlet_weight=0.25, return_policy=True)
            visit_sum += np.asarray(pt, dtype=np.float64)
        visits = visit_sum / visit_sum.sum()
        top_idx, pol_idx = int(visits.argmax()), _flat(pol)
        if top_idx == pol_idx:
            continue                                  # not a correction; skip
        order = np.argsort(-visits)[:a.topk_visits]
        out.append({'seed': seed, 'turn': int(fr['turn']), 'depth': depth,
                    'board': fr['board'], 'next_balls': fr['next_balls'],
                    'pol_idx': pol_idx, 'pol_share': float(visits[pol_idx]),
                    'mcts_top_idx': top_idx, 'mcts_top_share': float(visits[top_idx]),
                    'visits': [[int(j), float(visits[j])] for j in order if visits[j] > 0]})
    return out, n_states, max_depth


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--death-glob', default='crisis/death_games/death_*.json')
    p.add_argument('--lo', type=int, default=15)
    p.add_argument('--hi', type=int, default=85)       # D-15 .. D-85 (needs tail>=86 recordings)
    p.add_argument('--sims', type=int, default=4800)
    p.add_argument('--mcts-seeds', type=int, default=3)
    p.add_argument('--topk-visits', type=int, default=20)
    p.add_argument('--max-games', type=int, default=0)
    p.add_argument('--shards', type=int, default=1,
                   help='Split the death-game set into N shards for parallel processes.')
    p.add_argument('--shard', type=int, default=0,
                   help='This process handles shard index in [0, shards). Disjoint '
                        'per-game outputs => no collisions; run N procs / N fleet GPUs.')
    p.add_argument('--out-dir', default='crisis/corrections')
    a = p.parse_args()

    if a.shards > 1:
        torch.set_num_threads(1)   # multi-proc only: avoid core oversubscription. Left at
        # torch's default for single-proc so the baseline measures unconstrained.
    dev = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
    net, _ = load_model(MODEL, dev, fp16=(dev.type != 'cpu'))
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(a.death_glob))
    if a.shards > 1:
        files = [f for k, f in enumerate(files) if k % a.shards == a.shard]
    if a.max_games:
        files = files[:a.max_games]
    print(f"device={dev}  games={len(files)}  band depth[{a.lo},{a.hi}]  "
          f"MCTS@{a.sims} x{a.mcts_seeds} seeds  -> {a.out_dir}/", flush=True)

    tot_corr, tot_states, n_done, t0, truncated = 0, 0, 0, time.time(), 0
    for gi, f in enumerate(files):
        seed = int(re.search(r'_(\d+)\.json$', f).group(1))
        of = os.path.join(a.out_dir, f'corr_{seed}.json')
        if os.path.exists(of):                         # resume: skip done games
            continue
        try:
            g = json.load(open(f))
        except Exception:
            continue
        corr, n_states, max_depth = process_game(g, seed, net, dev, a)
        if max_depth < a.hi:
            truncated += 1                             # tail too short to reach D-hi
        json.dump({'seed': seed, 'n_band_states': n_states, 'max_depth': max_depth,
                   'corrections': corr}, open(of, 'w'), default=float)
        tot_corr += len(corr); tot_states += n_states; n_done += 1
        print(f"  [{gi+1}/{len(files)}] seed {seed}: {len(corr)}/{n_states} corrections "
              f"(running {tot_corr}/{tot_states}={100*tot_corr/max(tot_states,1):.0f}%, "
              f"{(time.time()-t0)/60:.1f}m)", flush=True)

    if truncated:
        print(f"WARNING: {truncated} games had tail < D-{a.hi} (re-record with longer tail "
              f"to use the full band).", flush=True)
    print(f"\nDONE this pass: {n_done} games, {tot_corr} corrections from {tot_states} "
          f"band states. Per-game files in {a.out_dir}/", flush=True)


if __name__ == '__main__':
    main()
