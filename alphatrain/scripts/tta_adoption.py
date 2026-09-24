"""Did a student learn its teacher's 8-view-averaged (TTA) decisions? On held-out states: where teacher TTA
differs from teacher single-view ('overrides'), how often does the student's single-view move equal the TTA
move (adoption); where they agree, how often does the student keep that move (keep)."""
import argparse, json, struct, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit
from alphatrain.scripts.tta_check import views, ACT
ap = argparse.ArgumentParser(); ap.add_argument('--states', required=True); ap.add_argument('--teacher', required=True); ap.add_argument('--students', nargs='+', required=True)
ap.add_argument('--n', type=int, default=10000); a = ap.parse_args()
recs = []
with open(a.states, 'rb') as f:
    f.read(4); n = struct.unpack('<i', f.read(4))[0]
    for _ in range(n):
        b = np.frombuffer(f.read(81), np.int8).reshape(9, 9).copy(); k = struct.unpack('<i', f.read(4))[0]
        nb = [struct.unpack('<iii', f.read(12)) for _ in range(3)][:k]; f.read(12); recs.append((b, nb))
sel = np.sort(np.random.default_rng(0).choice(len(recs), min(a.n, len(recs)), replace=False)); recs = [recs[i] for i in sel]; n = len(recs)
dev = torch.device('mps'); am = lambda b, lg: (lambda c, i, _: i[0] if c else -1)(*_legal_priors_jit(b, lg.astype(np.float32), 1))
def run(mp, tta):
    net, _ = load_model(mp, dev, fp16=False); out = np.zeros(n, np.int64); outv0 = np.zeros(n, np.int64)
    with torch.inference_mode():
        for s in range(0, n, 512):
            ch = recs[s:s+512]
            obs = torch.from_numpy(np.stack([o for b, nb in ch for o in (views(b, nb) if tta else views(b, nb)[:1])]).astype(np.float32))
            lg = net(obs.to(dev)).float().cpu().numpy().reshape(len(ch), 8 if tta else 1, 6561)
            for j in range(len(ch)):
                outv0[s + j] = am(ch[j][0], lg[j, 0])
                if tta: out[s + j] = am(ch[j][0], np.stack([lg[j, v, ACT[v]] for v in range(8)]).mean(0))
    return out, outv0
t_tta, t_v0 = run(a.teacher, True); over = t_tta != t_v0
print(f'{n:,} held-out states; teacher TTA overrides its own single view on {100*over.mean():.1f}%')
print(f'{"student":36s} {"adopt TTA overrides":>20s} {"keep agreed move":>17s} {"== teacher TTA overall":>23s}')
for mp in [a.teacher] + a.students:
    _, v0 = run(mp, False)
    print(f'{mp.split("/")[-1][:36]:36s} {100*(v0[over] == t_tta[over]).mean():19.1f}% {100*(v0[~over] == t_tta[~over]).mean():16.1f}% {100*(v0 == t_tta).mean():22.1f}%', flush=True)
