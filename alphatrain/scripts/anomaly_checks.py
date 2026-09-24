"""Anomaly hunt, part 1 (training-path correctness):
 A. training observation builder (TensorDatasetGPU._build_obs_core, MPS) vs reference build_observation (numba)
 B. legality of the stored target argmax in a training tensor (src occupied, dst empty, dst reachable)
 C. BatchNorm: argmax agreement of the net in train mode (batch stats) vs eval mode (running stats)"""
import argparse, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit

ap = argparse.ArgumentParser(); ap.add_argument('--tensor', default='alphatrain/data/r2_bulk.pt'); ap.add_argument('--model', default='alphatrain/data/scratch18b96_lr3e3_ckpts_epoch_40.pt')
ap.add_argument('--n', type=int, default=20000); a = ap.parse_args()
dev = torch.device('mps')
ds = TensorDatasetGPU(a.tensor, augment=False, color_augment=False, augment_factor=1, device='mps')
N = ds.boards.shape[0]; idx = torch.from_numpy(np.sort(np.random.default_rng(1).choice(N, a.n, replace=False))).to(dev)
B = ds.boards[idx]; NP = ds.next_pos[idx]; NC = ds.next_col[idx]; NN = ds.n_next[idx]
gpu = ds._build_obs_core(B, next_pos=NP, next_col=NC, n_next=NN).float().cpu().numpy()
Bn, NPn, NCn, NNn = B.cpu().numpy(), NP.cpu().numpy(), NC.cpu().numpy(), NN.cpu().numpy()
ref = np.stack([build_observation(Bn[i], NPn[i, :NNn[i], 0].astype(np.int64), NPn[i, :NNn[i], 1].astype(np.int64), NCn[i, :NNn[i]].astype(np.int64), int(NNn[i])) for i in range(a.n)])
print(f'A. training obs builder vs reference, {a.n:,} states:')
for ch in range(18):
    d = np.abs(gpu[:, ch] - ref[:, ch]); bad = (d > 1e-5).any(-1).any(-1)
    if bad.any(): print(f'   ch{ch:2d}: max|diff| {d.max():.4f}, states affected {100*bad.mean():.3f}%')
print('   (channels not listed are identical)')
pi = ds.pol_indices[idx].cpu().numpy(); pv = ds.pol_values[idx].float().cpu().numpy(); tgt = pi[np.arange(a.n), pv.argmax(1)]
ill = 0
for i in range(a.n):
    lg = np.full(6561, -1e9, np.float32); lg[tgt[i]] = 0.0
    c, j, _ = _legal_priors_jit(Bn[i], lg, 1); ill += int(c == 0 or j[0] != tgt[i])
print(f'B. stored target argmax illegal on its own board: {ill} / {a.n:,} ({100*ill/a.n:.3f}%)')
net, _ = load_model(a.model, dev, fp16=False)
x = torch.from_numpy(ref).to(dev)
def argmaxes(mode):
    net.train(mode); out = []
    with torch.no_grad():
        for s in range(0, a.n, 1024):
            lg = net(x[s:s+1024]).float().cpu().numpy()
            for j in range(lg.shape[0]):
                c, k, _ = _legal_priors_jit(Bn[s + j], lg[j], 1); out.append(k[0] if c else -1)
    net.train(False); return np.array(out)
ev, trm = argmaxes(False), argmaxes(True)
print(f'C. BN eval-mode vs train-mode (batch stats, bs 1024) argmax agreement: {100*(ev == trm).mean():.2f}%; '
      f'eval matches target {100*(ev == tgt).mean():.2f}%, train-mode matches target {100*(trm == tgt).mean():.2f}%')
