"""Select correction states (teacher argmax != base argmax) for the rollout judge.

Writes data/judge_states.bin:
  magic 'CLRJ', int32 N, per state:
    int8 board[81], int32 n_next, int32 (r,c,color) x3,
    int32 teacher_move, int32 base_move, float32 target_top_share

    python -m alphatrain.inference_cpp.export_judge_states \
        --tensor alphatrain/data/small128_iter1.pt \
        --base alphatrain/data/pillar3k_small128_hardce_epoch_87.pt --n 300
"""
import argparse, struct, sys
import numpy as np
import torch

sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', default='alphatrain/data/small128_iter1.pt')
    p.add_argument('--base', default='alphatrain/data/pillar3k_small128_hardce_epoch_87.pt')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--pool', type=int, default=40000, help='states to scan for corrections')
    p.add_argument('--out', default='alphatrain/inference_cpp/data/judge_states.bin')
    a = p.parse_args()

    d = torch.load(a.tensor, map_location='cpu', weights_only=False)
    rng = np.random.default_rng(2026)
    idx = rng.choice(d['boards'].shape[0], a.pool, replace=False)
    boards = d['boards'][idx].numpy()
    np_ = d['next_pos'][idx].numpy()
    nc = d['next_col'][idx].numpy()
    nn = d['n_next'][idx].numpy()
    pol_i = d['pol_indices'][idx].numpy()
    pol_v = d['pol_values'][idx].numpy()
    tgt_arg = pol_i[np.arange(a.pool), pol_v.argmax(1)]
    top = pol_v.max(1)

    dev = torch.device('mps')
    net, _ = load_model(a.base, dev, fp16=False)
    obs = np.zeros((a.pool, 18, 9, 9), dtype=np.float32)
    for i in range(a.pool):
        obs[i] = build_observation(boards[i], np_[i, :, 0].astype(np.int64),
                                   np_[i, :, 1].astype(np.int64),
                                   nc[i].astype(np.int64), int(nn[i]))
    obs_t = torch.from_numpy(obs)
    base_arg = np.zeros(a.pool, dtype=np.int64)
    with torch.inference_mode():
        for s in range(0, a.pool, 512):
            lg = net(obs_t[s:s+512].to(dev)).float().cpu().numpy()
            for j in range(lg.shape[0]):
                i = s + j
                k = pol_i[i][pol_v[i] > 0]
                base_arg[i] = k[lg[j][k].argmax()]

    dis = np.where(base_arg != tgt_arg)[0]
    print(f'pool {a.pool}: {len(dis)} corrections ({100*len(dis)/a.pool:.1f}%)')
    sel = rng.choice(dis, min(a.n, len(dis)), replace=False)

    with open(a.out, 'wb') as f:
        f.write(b'CLRJ')
        f.write(struct.pack('<i', len(sel)))
        for i in sel:
            f.write(boards[i].astype(np.int8).tobytes())
            f.write(struct.pack('<i', int(nn[i])))
            for t in range(3):
                f.write(struct.pack('<iii', int(np_[i, t, 0]), int(np_[i, t, 1]),
                                    int(nc[i, t])))
            f.write(struct.pack('<iif', int(tgt_arg[i]), int(base_arg[i]),
                                float(top[i])))
    print(f'wrote {a.out}: {len(sel)} correction states '
          f'(decisive>0.4 among them: {100*(top[sel]>0.4).mean():.0f}%)')


if __name__ == '__main__':
    main()
