"""Where do two checkpoints disagree, and who is right? Computes both models' greedy argmax on a CLRJ
state file, writes the disagreements as CLRJ (teacher = model A's move, base = model B's move) with
stratum meta, for rollout_judge (continuation policy chosen at judge time)."""
import argparse, json, struct, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit


def read_clrj(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'CLRJ'; n = struct.unpack('<i', f.read(4))[0]; recs = []
        for _ in range(n):
            board = f.read(81); k = struct.unpack('<i', f.read(4))[0]; nb = [struct.unpack('<iii', f.read(12)) for _ in range(3)]
            f.read(12); recs.append((board, k, nb))
    return recs


def argmaxes(model, recs, dev):
    net, _ = load_model(model, dev, fp16=False)
    obs = np.zeros((len(recs), 18, 9, 9), np.float32)
    for i, (b, k, nb) in enumerate(recs):
        obs[i] = build_observation(np.frombuffer(b, np.int8).reshape(9, 9), np.array([x[0] for x in nb[:k]]), np.array([x[1] for x in nb[:k]]), np.array([x[2] for x in nb[:k]]), k)
    out = np.zeros(len(recs), np.int64)
    with torch.inference_mode():
        for s in range(0, len(recs), 2048):
            lg = net(torch.from_numpy(obs[s:s+2048]).to(dev)).float().cpu().numpy()
            for j in range(lg.shape[0]):
                b = np.frombuffer(recs[s + j][0], np.int8).reshape(9, 9).copy()
                cnt, idx, _ = _legal_priors_jit(b, lg[j].astype(np.float32), 1)
                out[s + j] = int(idx[0]) if cnt > 0 else -1
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--states', required=True); p.add_argument('--a', required=True); p.add_argument('--b', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); recs = read_clrj(a.states); meta = json.load(open(a.states + '.meta.json')); dev = torch.device('mps')
    A = argmaxes(a.a, recs, dev); B = argmaxes(a.b, recs, dev); dis = np.where(A != B)[0]
    strat = np.array([m['stratum'] for m in meta])
    print(f'{len(recs)} states; A!=B on {len(dis)} ({100*len(dis)/len(recs):.1f}%); tail {100*(A!=B)[strat=="tail"].mean():.1f}% broad {100*(A!=B)[strat=="broad"].mean():.1f}%')
    with open(a.out, 'wb') as f:
        f.write(b'CLRJ'); f.write(struct.pack('<i', len(dis)))
        for i in dis:
            b, k, nb = recs[i]; f.write(b); f.write(struct.pack('<i', k))
            for t in range(3): f.write(struct.pack('<iii', *nb[t]))
            f.write(struct.pack('<iif', int(A[i]), int(B[i]), 0.0))
    json.dump([meta[int(i)] for i in dis], open(a.out + '.meta.json', 'w')); print('wrote', a.out)


if __name__ == '__main__':
    main()
