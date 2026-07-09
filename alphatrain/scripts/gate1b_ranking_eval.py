"""Gate 1b: does the survival head RANK candidate afterstates in agreement with
rollout truth? (ChatGPT review: calibration != useful move ordering.)

Inputs:
  --pairs   ranking_pairs.bin  (export_ranking_pairs.py: top1-vs-top2 per state)
  --results judge CSV for those pairs (rollout_judge --states ... --out ...)
  --head    trained value head (train_value_head.py output)
  --backbone ep87

For each pair: afterstate_i = board after applying move_i (with line clears),
same pending next_balls. Head scalar V = survival_to_scalar(head(features)).
Metric: pairwise accuracy of sign(V_top1 - V_top2) vs sign(rollout death gap)
on CLEAR pairs (|died_rate gap| > 0.08). Bar: >= 0.60.
"""
import argparse, struct, sys
import numpy as np
import torch

sys.path.insert(0, '.')
from game.board import _clear_lines_at
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
from alphatrain import value_head as vh


def load_pairs(path):
    out = []
    with open(path, 'rb') as f:
        assert f.read(4) == b'CLRJ'
        n = struct.unpack('<i', f.read(4))[0]
        for _ in range(n):
            board = np.frombuffer(f.read(81), dtype=np.int8).reshape(9, 9).copy()
            nn = struct.unpack('<i', f.read(4))[0]
            nb = []
            for t in range(3):
                r, c, col = struct.unpack('<iii', f.read(12))
                if t < nn:
                    nb.append((r, c, col))
            m1, m2, ts = struct.unpack('<iif', f.read(12))
            out.append((board, nb, m1, m2))
    return out


def afterstate(board, mv):
    b = board.copy()
    src, tgt = mv // 81, mv % 81
    sr, sc, tr, tc = src // 9, src % 9, tgt // 9, tgt % 9
    color = b[sr, sc]
    b[sr, sc] = 0
    b[tr, tc] = color
    _clear_lines_at(b, tr, tc)  # in place
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', default='alphatrain/inference_cpp/data/ranking_pairs.bin')
    p.add_argument('--results', default='alphatrain/inference_cpp/data/ranking_results.csv')
    p.add_argument('--head', default='alphatrain/data/value_head_small128.pt')
    p.add_argument('--backbone', default='alphatrain/data/pillar3k_small128_hardce_epoch_87.pt')
    p.add_argument('--clear-gap', type=float, default=0.08)
    a = p.parse_args()

    pairs = load_pairs(a.pairs)
    res = np.genfromtxt(a.results, delimiter=',', names=True)
    assert len(pairs) == len(res), f'{len(pairs)} pairs vs {len(res)} results'

    dev = torch.device('mps')
    net, _ = load_model(a.backbone, dev, fp16=False)
    head, ckpt, head_type = vh.load_any(a.head, dev)
    head.train(False)
    print(f'head type={head_type} target={ckpt.get("target_type")}')

    def head_v(board, nb):
        nr = np.zeros(3, dtype=np.int64); nc = np.zeros(3, dtype=np.int64)
        ncol = np.zeros(3, dtype=np.int64)
        for t, (r, c, col) in enumerate(nb[:3]):
            nr[t], nc[t], ncol[t] = r, c, col
        obs = build_observation(board, nr, nc, ncol, min(len(nb), 3))
        with torch.inference_mode():
            feats = net.backbone_features(
                torch.from_numpy(obs).unsqueeze(0).to(dev))
            out = head(feats.float())
            if head_type == 'spatial':
                return float(out.squeeze())
            return float(vh.survival_to_scalar(out).squeeze())

    dv, gap = [], []
    for i, (board, nb, m1, m2) in enumerate(pairs):
        v1 = head_v(afterstate(board, m1), nb)
        v2 = head_v(afterstate(board, m2), nb)
        dv.append(v1 - v2)
        # judge arm0 = m1 ('teacher' column), arm1 = m2 ('base' column)
        gap.append(float(res['base_died'][i] - res['teacher_died'][i]))  # >0: m1 better
    dv, gap = np.array(dv), np.array(gap)

    clear = np.abs(gap) > a.clear_gap
    n_clear = int(clear.sum())
    acc = float((np.sign(dv[clear]) == np.sign(gap[clear])).mean()) if n_clear else float('nan')
    rho = float(np.corrcoef(dv, gap)[0, 1])
    print(f'pairs: {len(dv)}  clear-gap pairs (|gap|>{a.clear_gap}): {n_clear}')
    print(f'PAIRWISE RANKING ACCURACY on clear pairs: {acc:.3f}  (bar: >=0.60)')
    print(f'corr(head dV, rollout gap) over all pairs: {rho:.3f}')
    print('GATE 1b:', 'PASS' if (n_clear >= 20 and acc >= 0.60) else 'FAIL/INSUFFICIENT')


if __name__ == '__main__':
    main()
