"""Where do rotated views disagree: on the SOURCE ball, the DESTINATION cell, or both?"""
import struct, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit
from alphatrain.scripts.tta_check import views, ACT
st, mp = sys.argv[1], sys.argv[2]
recs = []
with open(st, 'rb') as f:
    f.read(4); n = struct.unpack('<i', f.read(4))[0]
    for _ in range(n):
        b = np.frombuffer(f.read(81), np.int8).reshape(9, 9).copy(); k = struct.unpack('<i', f.read(4))[0]
        nb = [struct.unpack('<iii', f.read(12)) for _ in range(3)][:k]; f.read(12); recs.append((b, nb))
net, _ = load_model(mp, torch.device('mps'), fp16=False); A = np.zeros((n, 8), np.int64)
with torch.inference_mode():
    for s in range(0, n, 512):
        ch = recs[s:s+512]; obs = torch.from_numpy(np.stack([o for b, nb in ch for o in views(b, nb)]).astype(np.float32))
        lg = net(obs.to('mps')).float().cpu().numpy().reshape(len(ch), 8, 6561)
        for j in range(len(ch)):
            for v in range(8):
                c, i, _ = _legal_priors_jit(ch[j][0], lg[j, v, ACT[v]].astype(np.float32), 1); A[s + j, v] = i[0] if c else -1
src, dst = A // 81, A % 81
dif = A[:, 1:] != A[:, :1]; ds = src[:, 1:] != src[:, :1]; dd = dst[:, 1:] != dst[:, :1]
print(f'{mp.split("/")[-1]}: view k != view 0 on {100*dif.mean():.1f}% of (state, view) pairs; of those: '
      f'same destination/different ball {100*(ds & ~dd).sum()/dif.sum():.1f}%, same ball/different destination {100*(~ds & dd).sum()/dif.sum():.1f}%, both differ {100*(ds & dd).sum()/dif.sum():.1f}%')
