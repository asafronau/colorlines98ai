"""Color-relabeling diagnostics on recorded states (eval --record-dir game_seed*.json), HISTORY 279.

1. Reference check for `eval --color-tta K [--tta 8]`: recompute the view-averaged legal argmax in fp32 Python
   (D4 view v = j // K, color shift k = j % K, c -> (c - 1 + k) % 7 + 1, logits mapped back by the view's action
   map, mean) and compare it with the move the C++ engine recorded.
2. How much symmetry the model has learned: for each of the 7 non-trivial D4 views and the 6 non-trivial cyclic
   color relabelings, the share of states whose legal argmax equals the plain one, and the share where all agree.

    python -m alphatrain.scripts.color_tta_check --games-dir DIR --model alphatrain/data/ta_A7_e4_a1.0.pt \
        --color-tta 7 [--tta 8]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit
from alphatrain.observation import build_observation

M = 8
F = [lambda r, c: (r, c), lambda r, c: (c, M - r), lambda r, c: (M - r, M - c), lambda r, c: (M - c, r),
     lambda r, c: (r, M - c), lambda r, c: (M - r, c), lambda r, c: (c, r), lambda r, c: (M - c, M - r)]
CELL = np.array([[f(r, c)[0] * 9 + f(r, c)[1] for r in range(9) for c in range(9)] for f in F])
ACT = (CELL[:, :, None] * 81 + CELL[:, None, :]).reshape(8, 6561)


def relabel(c, k):
    return 0 if c == 0 else (c - 1 + k) % 7 + 1


def view_obs(board, nb, v, k):
    lut = np.array([relabel(c, k) for c in range(8)], np.int8)
    b2 = np.zeros(81, np.int8); b2[CELL[v]] = lut[board.reshape(81)]
    rr = np.array([F[v](r, c)[0] for r, c, _ in nb], np.int64)
    cc = np.array([F[v](r, c)[1] for r, c, _ in nb], np.int64)
    col = np.array([relabel(x, k) for _, _, x in nb], np.int64)
    return build_observation(b2.reshape(9, 9), rr, cc, col, len(nb))


def logits_for(net, dev, states, views):
    """(n, len(views), 6561) logits mapped back to each state's original frame."""
    obs = np.stack([view_obs(b, nb, v, k) for b, nb, _ in states for v, k in views])
    out = np.zeros((len(obs), 6561), np.float32)
    with torch.inference_mode():
        for s in range(0, len(obs), 4096):
            out[s:s + 4096] = net(torch.from_numpy(obs[s:s + 4096]).to(dev)).float().cpu().numpy()
    out = out.reshape(len(states), len(views), 6561)
    return np.stack([out[:, j, ACT[v]] for j, (v, _) in enumerate(views)], 1)


def legal_argmax(board, logits):
    c, idx, _ = _legal_priors_jit(board, logits.astype(np.float32).copy(), 1)
    return int(idx[0]) if c else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games-dir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--color-tta', type=int, default=7)
    ap.add_argument('--tta', type=int, default=1, choices=[1, 8])
    ap.add_argument('--max-states', type=int, default=4000)
    a = ap.parse_args()
    states = []
    for f in sorted(glob.glob(os.path.join(a.games_dir, 'game_seed*.json'))):
        for s in json.load(open(f))['states']:
            states.append((np.array(s['board'], np.int8).reshape(9, 9),
                           [(x['row'], x['col'], x['color']) for x in s['next_balls']], int(s['move'])))
    states = states[:a.max_states]
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net, _ = load_model(a.model, dev, fp16=False)
    n = len(states)

    views = [(j // a.color_tta, j % a.color_tta) for j in range(a.tta * a.color_tta)]
    L = logits_for(net, dev, states, views).mean(1)
    ref = np.array([legal_argmax(b, L[i]) for i, (b, _, _) in enumerate(states)])
    rec = np.array([mv for _, _, mv in states])
    print(f'{n} recorded states; fp32 Python reference of tta {a.tta} x color-tta {a.color_tta} == C++ move: '
          f'{100 * (ref == rec).mean():.2f}%', flush=True)

    Ld = logits_for(net, dev, states, [(v, 0) for v in range(8)])
    perd = np.array([[legal_argmax(b, Ld[i, v]) for v in range(8)] for i, (b, _, _) in enumerate(states)])
    samed = perd[:, 1:] == perd[:, :1]
    print('plain argmax kept under D4 views 1..7:        ' + ' '.join(f'{100 * samed[:, v].mean():.1f}%' for v in range(7))
          + f' | all 8 agree {100 * samed.all(1).mean():.1f}%')
    Lc = logits_for(net, dev, states, [(0, k) for k in range(7)])
    per = np.array([[legal_argmax(b, Lc[i, k]) for k in range(7)] for i, (b, _, _) in enumerate(states)])
    same = per[:, 1:] == per[:, :1]
    print('plain argmax kept under relabeling k=1..6: ' + ' '.join(f'{100 * same[:, k].mean():.1f}%' for k in range(6))
          + f' | all 7 agree {100 * same.all(1).mean():.1f}% | 7-relabel average != plain '
          f'{100 * (np.array([legal_argmax(b, Lc[i].mean(0)) for i, (b, _, _) in enumerate(states)]) != per[:, 0]).mean():.1f}%')


if __name__ == '__main__':
    main()
