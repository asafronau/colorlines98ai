"""Global decisiveness of checkpoints on a CLRJ state file: legal-softmax top-1 prob, top1-top2 logit margin, entropy over legal moves."""
import argparse, struct, sys, numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit
p = argparse.ArgumentParser(); p.add_argument('--states', required=True); p.add_argument('--models', nargs='+', required=True); a = p.parse_args()
with open(a.states, 'rb') as f:
    assert f.read(4) == b'CLRJ'; n = struct.unpack('<i', f.read(4))[0]; recs = []
    for _ in range(n):
        b = np.frombuffer(f.read(81), np.int8).reshape(9, 9).copy(); k = struct.unpack('<i', f.read(4))[0]; nb = [struct.unpack('<iii', f.read(12)) for _ in range(3)]; f.read(12); recs.append((b, k, nb))
obs = np.zeros((n, 18, 9, 9), np.float32)
for i, (b, k, nb) in enumerate(recs): obs[i] = build_observation(b, np.array([x[0] for x in nb[:k]]), np.array([x[1] for x in nb[:k]]), np.array([x[2] for x in nb[:k]]), k)
dev = torch.device('mps'); ot = torch.from_numpy(obs)
print(f'{"model":44s} {"top1 P50":>8s} {"margin P50":>10s} {"entropy P50":>11s} {"logit std":>9s}')
for m in a.models:
    net, _ = load_model(m, dev, fp16=False); top1 = []; marg = []; ent = []; lstd = []
    with torch.inference_mode():
        for s in range(0, n, 2048):
            lg = net(ot[s:s+2048].to(dev)).float().cpu().numpy()
            for j in range(lg.shape[0]):
                cnt, idx, pri = _legal_priors_jit(recs[s + j][0], lg[j].astype(np.float32), 30)
                pri = pri[:cnt]; top1.append(pri.max()); ent.append(-(pri * np.log(pri + 1e-12)).sum())
                srt = np.sort(lg[j][idx[:cnt]])[::-1]; marg.append(srt[0] - srt[1] if cnt > 1 else 0); lstd.append(lg[j][idx[:cnt]].std())
    print(f'{m.split("/")[-1]:44s} {np.median(top1):8.3f} {np.median(marg):10.2f} {np.median(ent):11.3f} {np.median(lstd):9.2f}')
