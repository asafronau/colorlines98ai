"""Phase 0b (redesigned): export TRUE student-visited anchor states for the judge.

The crisis tensors are ~99.7% MCTS-replay states; only the FIRST state of each
replay game is a state the greedy student actually visited. This script walks the
crisis games dir, takes each game's first state (the anchor), computes the FULL
legal argmax of both the student and the teacher (no support restriction — the
old exporter's base move was restricted to the stored teacher top-5), and writes:

  1. an audit table (agreement by label, teacher logit-gap strata) — the
     corrected Phase 0a numbers,
  2. judge_states.bin (CLRJ format) for the DISAGREEMENT anchors:
     teacher_move = teacher full-legal argmax, base_move = student full-legal
     argmax, float field = teacher logit gap (t_logit[t_arg] - t_logit[s_arg]),
  3. a metadata sidecar CSV (same order as the .bin): original_seed, label,
     turns_at_anchor, gap — for cluster bootstrap + strata joins.

    python -m alphatrain.scripts.export_dagger_anchors \
        --games-dir data/crisis_cpp128_v1 \
        --student alphatrain/data/small128_vh1.pt \
        --teacher alphatrain/data/pillar3k_r3_dw3_T0.7_epoch_22.pt \
        --out-bin alphatrain/inference_cpp/data/dagger_judge_states.bin \
        --out-meta alphatrain/inference_cpp/data/dagger_judge_meta.csv
"""
import argparse
import glob
import json
import os
import struct

import numpy as np
import torch

from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit


def anchor_from_game(path):
    with open(path) as f:
        d = json.load(f)
    m0 = d['moves'][0]
    board = np.array(m0['board'], dtype=np.int8).reshape(81)
    nb = [(b['row'], b['col'], b['color']) for b in m0['next_balls']]
    return {
        'board': board,
        'nb': nb,
        'label': d['label'],
        'original_seed': int(d['original_seed']),
        'turns': int(d.get('replay_from_turn', -1)),
    }


def full_legal_argmax(net, dtype, dev, boards, nbs, batch=512):
    """Return (argmax flat move, raw logits) per state, argmax over ALL legal moves."""
    n = len(boards)
    obs = np.zeros((n, 18, 9, 9), dtype=np.float32)
    for i in range(n):
        rows = np.array([b[0] for b in nbs[i]], dtype=np.int64)
        cols = np.array([b[1] for b in nbs[i]], dtype=np.int64)
        colv = np.array([b[2] for b in nbs[i]], dtype=np.int64)
        obs[i] = build_observation(boards[i].reshape(9, 9), rows, cols, colv,
                                   len(nbs[i]))
    args_out = np.full(n, -1, dtype=np.int64)
    logits_out = np.zeros((n, 6561), dtype=np.float32)
    obs_t = torch.from_numpy(obs)
    with torch.inference_mode():
        for s in range(0, n, batch):
            lg = net(obs_t[s:s + batch].to(dev).to(dtype)).float().cpu().numpy()
            for j in range(lg.shape[0]):
                i = s + j
                k, fi, _ = _legal_priors_jit(boards[i].reshape(9, 9), lg[j], 1)
                if k > 0:
                    args_out[i] = int(fi[0])
                logits_out[i] = lg[j]
    return args_out, logits_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', default='data/crisis_cpp128_v1')
    p.add_argument('--student', default='alphatrain/data/small128_vh1.pt')
    p.add_argument('--teacher',
                   default='alphatrain/data/pillar3k_r3_dw3_T0.7_epoch_22.pt')
    p.add_argument('--out-bin',
                   default='alphatrain/inference_cpp/data/dagger_judge_states.bin')
    p.add_argument('--out-meta',
                   default='alphatrain/inference_cpp/data/dagger_judge_meta.csv')
    p.add_argument('--device', default='mps')
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(a.games_dir, 'game_*.json')))
    print(f'{len(files)} replay games in {a.games_dir}', flush=True)
    anchors = []
    for i, f in enumerate(files):
        anchors.append(anchor_from_game(f))
        if (i + 1) % 2000 == 0:
            print(f'  parsed {i + 1}/{len(files)}', flush=True)
    seeds = {x['original_seed'] for x in anchors}
    print(f'{len(anchors)} anchors from {len(seeds)} independent deaths '
          f'(prevention {sum(1 for x in anchors if x["label"] == "prevention")}, '
          f'recovery {sum(1 for x in anchors if x["label"] == "recovery")})',
          flush=True)

    dev = torch.device(a.device)
    boards = [x['board'] for x in anchors]
    nbs = [x['nb'] for x in anchors]
    s_net, _ = load_model(a.student, dev, fp16=False)
    s_arg, _ = full_legal_argmax(s_net, torch.float32, dev, boards, nbs)
    del s_net
    t_net, _ = load_model(a.teacher, dev, fp16=False)
    t_arg, t_logits = full_legal_argmax(t_net, torch.float32, dev, boards, nbs)
    del t_net

    valid = (s_arg >= 0) & (t_arg >= 0)
    gap = np.zeros(len(anchors), dtype=np.float32)
    idx = np.arange(len(anchors))
    gap[valid] = (t_logits[idx[valid], t_arg[valid]]
                  - t_logits[idx[valid], s_arg[valid]])

    # ---- audit table (corrected Phase 0a) ----
    print('\n=== corrected 0a: TRUE student-visited anchors ===', flush=True)
    for lab in ('all', 'prevention', 'recovery'):
        m = valid if lab == 'all' else (
            valid & np.array([x['label'] == lab for x in anchors]))
        agree = (s_arg[m] == t_arg[m])
        dis = m & (s_arg != t_arg)
        g = gap[dis]
        print(f'{lab:11s}: n={m.sum():5d}  top1-agree {100 * agree.mean():5.2f}%  '
              f'| disagreements {dis.sum():4d}: gap median {np.median(g):.2f}  '
              f'>=1.0 {100 * (g >= 1.0).mean():4.1f}%', flush=True)

    # ---- judge .bin + sidecar for disagreements ----
    sel = np.where(valid & (s_arg != t_arg))[0]
    with open(a.out_bin, 'wb') as f:
        f.write(b'CLRJ')
        f.write(struct.pack('<i', len(sel)))
        for i in sel:
            f.write(anchors[i]['board'].astype(np.int8).tobytes())
            nb = anchors[i]['nb']
            f.write(struct.pack('<i', len(nb)))
            for t in range(3):
                r, c, col = nb[t] if t < len(nb) else (0, 0, 0)
                f.write(struct.pack('<iii', r, c, col))
            f.write(struct.pack('<iif', int(t_arg[i]), int(s_arg[i]),
                                float(gap[i])))
    with open(a.out_meta, 'w') as f:
        f.write('state,original_seed,label,turns,gap\n')
        for row, i in enumerate(sel):
            f.write(f'{row},{anchors[i]["original_seed"]},{anchors[i]["label"]},'
                    f'{anchors[i]["turns"]},{gap[i]:.4f}\n')
    print(f'\nwrote {a.out_bin} + {a.out_meta}: {len(sel)} disagreement anchors '
          f'({100 * len(sel) / valid.sum():.1f}% of valid)', flush=True)


if __name__ == '__main__':
    main()
