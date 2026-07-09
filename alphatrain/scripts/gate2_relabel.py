"""Gate 2: re-search sampled states with ep87 + the new value head @600 sims
(no Dirichlet, temp=0) across a q_weight sweep; write each q's search-vs-base
DISAGREEMENTS in the judge format for rollout adjudication.

    PYTHONPATH=. python -m alphatrain.scripts.gate2_relabel --n 400
"""
import argparse, glob, json, struct, sys
import numpy as np
import torch

sys.path.insert(0, '.')
from game.board import ColorLinesGame
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
from alphatrain.mcts import MCTS, _legal_priors_jit


def sample_states(games_dir, n, rng):
    files = sorted(glob.glob(f'{games_dir}/game_seed*.json'))
    rng.shuffle(files)
    out = []
    for f in files:
        d = json.load(open(f))
        mv = d['moves']
        if len(mv) < 10:
            continue
        picks = {max(0, len(mv) - 1 - int(rng.integers(0, 40))),
                 int(rng.integers(0, len(mv)))}
        for i in picks:
            out.append(mv[i])
        if len(out) >= n:
            break
    return out[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3k_small128_hardce_epoch_87.pt')
    p.add_argument('--head', default='alphatrain/data/value_head_small128.pt')
    p.add_argument('--games-dir', default='data/crisis_cpp128_v1_died')
    p.add_argument('--n', type=int, default=400)
    p.add_argument('--sims', type=int, default=600)
    p.add_argument('--q-values', type=float, nargs='+', default=[0.0, 1.0, 2.0, 3.0])
    p.add_argument('--out-prefix', default='alphatrain/inference_cpp/data/gate2')
    a = p.parse_args()

    rng = np.random.default_rng(4242)
    states = sample_states(a.games_dir, a.n, rng)
    print(f'{len(states)} states sampled from {a.games_dir}')

    dev = torch.device('mps')
    net, max_score = load_model(a.model, dev, fp16=False)

    # base argmax per state
    prepared = []
    for m in states:
        board = np.array(m['board'], dtype=np.int8)
        nb = m['next_balls']
        nr = np.zeros(3, dtype=np.int64); nc = np.zeros(3, dtype=np.int64)
        ncol = np.zeros(3, dtype=np.int64)
        nn = min(len(nb), 3)
        for t in range(nn):
            nr[t], nc[t], ncol[t] = nb[t]['row'], nb[t]['col'], nb[t]['color']
        obs = build_observation(board, nr, nc, ncol, nn)
        with torch.inference_mode():
            lg = net(torch.from_numpy(obs).unsqueeze(0).to(dev)).float().cpu().numpy()[0]
        k, fi, pr = _legal_priors_jit(board, lg, 1)
        if k == 0:
            continue
        base = int(fi[np.argmax(pr[:k])])
        prepared.append((board, nb, nn, base))
    print(f'{len(prepared)} states with legal moves')

    for q in a.q_values:
        mcts = MCTS(net=net, device=dev, max_score=max_score,
                    num_simulations=a.sims, c_puct=2.5, top_k=30, batch_size=8,
                    q_weight=q, value_head_path=a.head)
        recs = []
        for si, (board, nb, nn, base) in enumerate(prepared):
            g = ColorLinesGame(seed=1)
            g.reset(board=board.copy(),
                    next_balls=[((b['row'], b['col']), b['color']) for b in nb])
            mv = mcts.search(g, temperature=0.0)  # no Dirichlet by default
            if mv is None:
                continue
            (sr, sc), (tr, tc) = mv
            search_flat = (sr * 9 + sc) * 81 + (tr * 9 + tc)
            if search_flat != base:
                recs.append((board, nb, nn, search_flat, base))
            if (si + 1) % 100 == 0:
                print(f'  q={q}: {si+1}/{len(prepared)} searched, '
                      f'{len(recs)} disagreements', flush=True)

        out = f'{a.out_prefix}_q{q:g}.bin'
        with open(out, 'wb') as f:
            f.write(b'CLRJ')
            f.write(struct.pack('<i', len(recs)))
            for board, nb, nn, search_flat, base in recs:
                f.write(board.astype(np.int8).tobytes())
                f.write(struct.pack('<i', nn))
                for t in range(3):
                    if t < nn:
                        f.write(struct.pack('<iii', nb[t]['row'], nb[t]['col'],
                                            nb[t]['color']))
                    else:
                        f.write(struct.pack('<iii', 0, 0, 0))
                f.write(struct.pack('<iif', search_flat, base, 0.0))
        print(f'q={q}: {len(recs)}/{len(prepared)} disagreements '
              f'({100*len(recs)/len(prepared):.1f}%) -> {out}', flush=True)


if __name__ == '__main__':
    main()
