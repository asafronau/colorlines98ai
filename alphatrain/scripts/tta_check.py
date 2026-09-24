"""Symmetry test-time averaging (TTA) diagnostic. The game is exactly D4-symmetric; the net is not.
For each state, build the observation of all 8 dihedral views from the TRANSFORMED board/preview
(no channel permutation tricks), run the net, map logits back to the original frame, average,
take the legal argmax. Reports per-view consistency and how the TTA argmax relates to the search's
robust / noise overrides and agree rows (same classes as override_consistency.py)."""
import argparse, json, struct, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit

M = 8
F = [lambda r, c: (r, c), lambda r, c: (c, M - r), lambda r, c: (M - r, M - c), lambda r, c: (M - c, r),
     lambda r, c: (r, M - c), lambda r, c: (M - r, c), lambda r, c: (c, r), lambda r, c: (M - c, M - r)]
CELL = np.array([[f(r, c)[0] * 9 + f(r, c)[1] for r in range(9) for c in range(9)] for f in F])   # (8, 81): orig cell -> view cell
ACT = (CELL[:, :, None] * 81 + CELL[:, None, :]).reshape(8, 6561)                                  # orig action -> view action


def views(board, nb):
    out = []
    for v, f in enumerate(F):
        b2 = np.zeros(81, np.int8); b2[CELL[v]] = board.reshape(81)
        rr = np.array([f(x[0], x[1])[0] for x in nb], np.int64); cc = np.array([f(x[0], x[1])[1] for x in nb], np.int64)
        col = np.array([x[2] for x in nb], np.int64)
        out.append(build_observation(b2.reshape(9, 9), rr, cc, col, len(nb)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--states', required=True); ap.add_argument('--models', nargs='+', required=True); a = ap.parse_args()
    meta = json.load(open(a.states + '.meta.json')); n = len(meta)
    chosen = np.array([m['chosen'] for m in meta]); over = np.array([m['over'] for m in meta])
    robust = np.load(a.states + '.robust.npy'); noise = np.load(a.states + '.noise.npy')
    recs = []
    with open(a.states, 'rb') as f:
        f.read(8)
        for _ in range(n):
            b = np.frombuffer(f.read(81), np.int8).reshape(9, 9).copy(); k = struct.unpack('<i', f.read(4))[0]
            nb = [struct.unpack('<iii', f.read(12)) for _ in range(3)][:k]; f.read(12); recs.append((b, nb))
    obs = torch.from_numpy(np.stack([o for b, nb in recs for o in views(b, nb)]).astype(np.float32))   # (n*8, 18, 9, 9)
    dev = torch.device('mps')
    print(f'{n} states; robust {robust.sum()}, noise {noise.sum()}, agree {(~over).sum()}')
    print(f'{"model":34s} {"mode":6s} {"adopt ROBUST":>12s} {"adopt NOISE":>11s} {"keep AGREE":>10s} {"all-8-views agree":>17s} {"TTA!=view0":>10s}')
    for mp in a.models:
        net, _ = load_model(mp, dev, fp16=False); L = np.zeros((n, 8, 6561), np.float32)
        with torch.inference_mode():
            for s in range(0, n * 8, 4096):
                L.reshape(n * 8, 6561)[s:s+4096] = net(obs[s:s+4096].to(dev)).float().cpu().numpy()
        back = np.stack([L[:, v, ACT[v]] for v in range(8)], 1)       # (n, 8, 6561) in original frame
        per_view = np.zeros((n, 8), np.int64); tta = np.zeros(n, np.int64)
        avg = back.mean(1)
        for i in range(n):
            b = recs[i][0]
            for v in range(8):
                c, idx, _ = _legal_priors_jit(b, back[i, v].copy(), 1); per_view[i, v] = idx[0] if c else -1
            c, idx, _ = _legal_priors_jit(b, avg[i].copy(), 1); tta[i] = idx[0] if c else -1
        all8 = (per_view == per_view[:, :1]).all(1)
        for mode, arg in (('view0', per_view[:, 0]), ('TTA-8', tta)):
            hit = arg == chosen
            print(f'{mp.split("/")[-1][:34]:34s} {mode:6s} {100*hit[robust].mean():12.1f} {100*hit[noise].mean():11.1f} {100*hit[~over].mean():10.1f} '
                  f'{100*all8.mean():16.1f}% {100*(tta != per_view[:, 0]).mean():9.1f}%', flush=True)


if __name__ == '__main__':
    main()
