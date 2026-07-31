"""Gate 1b: build afterstate-ranking pairs for the rollout judge.

Samples states from died games across the danger spectrum, takes ep87's TOP-2
legal policy moves (the realistic search dilemma), and writes them in the
judge_states.bin format (moveA=top1 -> 'teacher_move', moveB=top2 ->
'base_move'). The rollout judge then labels each pair with died-within-H rates;
gate 1b checks the value head ranks the two AFTERSTATES in agreement with the
rollout truth on clear-gap pairs (bar: >=0.60 pairwise accuracy).

    python -m alphatrain.inference_cpp.export_ranking_pairs --n 400
"""
import argparse, glob, json, struct, sys
import numpy as np
import torch

sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3k_small128_hardce_epoch_87.pt')
    p.add_argument('--games-dir', default='data/crisis_cpp128_v1_died')
    p.add_argument('--n', type=int, default=400)
    p.add_argument('--out', default='alphatrain/inference_cpp/data/ranking_pairs.bin')
    a = p.parse_args()

    rng = np.random.default_rng(31337)
    files = sorted(glob.glob(f'{a.games_dir}/game_seed*.json'))
    rng.shuffle(files)

    # Sample states biased toward the death neighborhood: for each died game,
    # take one state from the last 40 moves and one uniformly.
    states = []
    for f in files:
        d = json.load(open(f))
        mv = d['moves']
        if len(mv) < 10:
            continue
        picks = {max(0, len(mv) - 1 - int(rng.integers(0, 40))),
                 int(rng.integers(0, len(mv)))}
        for i in picks:
            m = mv[i]
            states.append(m)
        if len(states) >= a.n:
            break
    states = states[:a.n]

    dev = torch.device('mps')
    net, _ = load_model(a.model, dev, fp16=False)

    recs = []
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
        k, fi, pr = _legal_priors_jit(board, lg, 2)
        if k < 2:
            continue
        order = np.argsort(-pr[:k])
        top1, top2 = int(fi[order[0]]), int(fi[order[1]])
        recs.append((board, nb, nn, top1, top2))

    with open(a.out, 'wb') as f:
        f.write(b'CLRJ')
        f.write(struct.pack('<i', len(recs)))
        for board, nb, nn, top1, top2 in recs:
            f.write(board.astype(np.int8).tobytes())
            f.write(struct.pack('<i', nn))
            for t in range(3):
                if t < nn:
                    f.write(struct.pack('<iii', nb[t]['row'], nb[t]['col'], nb[t]['color']))
                else:
                    f.write(struct.pack('<iii', 0, 0, 0))
            f.write(struct.pack('<iif', top1, top2, 0.0))
    print(f'wrote {a.out}: {len(recs)} top1-vs-top2 pairs from died games')


if __name__ == '__main__':
    main()
