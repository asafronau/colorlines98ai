"""Relabel every row of a train_path_b tensor with a policy's 8-view-averaged (TTA-8) decision, on the GPU:
obs built by the training builder, rotated with the verified dataset LUTs, logits mapped back with the policy
LUTs, averaged, legal-masked, top-5 kept. The model is loaded exactly as deployed (fp16 + fp16-safe BN).
Writes an .npz sidecar: tta_idx/tta_val (top-5, softmax over the 5), view0_arg, orig_arg, orig_top."""
import argparse, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.dataset import TensorDatasetGPU, _transform_observation
from alphatrain.scripts.fleet_gpu import gpu_legal_mask
from alphatrain.inference_cpp.export_ts import load_model as load_deploy_model
from alphatrain.mcts import _legal_priors_jit


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tensor', required=True); p.add_argument('--model', required=True)
    p.add_argument('--out', required=True); p.add_argument('--chunk', type=int, default=4096); p.add_argument('--limit', type=int, default=0)
    a = p.parse_args(); dev = torch.device('mps')
    ds = TensorDatasetGPU(a.tensor, augment=False, color_augment=False, augment_factor=1, device='mps')
    net, _, _ = load_deploy_model(a.model, dev, torch.float16)
    N = ds.boards.shape[0] if not a.limit else a.limit
    obs_inv = [ds._obs_inv_luts[t] for t in range(8)]; line_inv = [ds._line_inv_luts[t] for t in range(8)]
    pol = [torch.as_tensor(ds._pol_luts[t], device=dev).long() for t in range(8)]
    tta_idx = np.zeros((N, 5), np.int64); tta_val = np.zeros((N, 5), np.float16); v0 = np.zeros(N, np.int64)
    oi = ds.pol_indices[:N].cpu().numpy(); ov = ds.pol_values[:N].float().cpu().numpy()
    orig_arg = oi[np.arange(N), ov.argmax(1)]; orig_top = ov.max(1)
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, N, a.chunk):
            e = min(N, s + a.chunk); idx = torch.arange(s, e, device=dev)
            B = ds.boards[idx]
            obs0 = ds._build_obs_core(B, next_pos=ds.next_pos[idx], next_col=ds.next_col[idx], n_next=ds.n_next[idx]).float()
            acc = None
            for t in range(8):
                ob = obs0 if t == 0 else _transform_observation(obs0, obs_inv[t], line_inv[t])
                lg = net(ob.half()).float()
                back = lg[:, pol[t]]                                   # original action a <- view action pol[t][a]
                if t == 0: v0_lg = back
                acc = back if acc is None else acc + back
            legal = gpu_legal_mask(B)
            acc = acc.masked_fill(~legal, float('-inf')); v0_lg = v0_lg.masked_fill(~legal, float('-inf'))
            top = acc.topk(5, dim=1); vals = torch.softmax(top.values, dim=1)
            vals = torch.nan_to_num(vals, nan=0.0)
            tta_idx[s:e] = top.indices.cpu().numpy(); tta_val[s:e] = vals.cpu().numpy().astype(np.float16)
            v0[s:e] = v0_lg.argmax(1).cpu().numpy()
            if (s // a.chunk) % 200 == 0:
                el = time.time() - t0; print(f'  {e:,}/{N:,}  {e/el:.0f} states/s  ETA {(N-e)/(e/el)/60:.1f} min', flush=True)
    # exact legality check of the TTA argmax on a CPU sample (GPU legal mask is a fast labeller)
    rng = np.random.default_rng(0); samp = rng.choice(N, min(20000, N), replace=False); bad = 0
    Bn = ds.boards[torch.as_tensor(samp, device=dev)].cpu().numpy()
    for j, i in enumerate(samp):
        lg = np.full(6561, -1e9, np.float32); lg[tta_idx[i, 0]] = 0.0
        c, k, _ = _legal_priors_jit(Bn[j], lg, 1); bad += int(c == 0 or k[0] != tta_idx[i, 0])
    np.savez(a.out, tta_idx=tta_idx, tta_val=tta_val, view0_arg=v0, orig_arg=orig_arg, orig_top=orig_top)
    tta_arg = tta_idx[:, 0]
    print(f'wrote {a.out}: {N:,} rows in {(time.time()-t0)/60:.1f} min; TTA argmax illegal on {bad}/{len(samp)} sampled rows')
    print(f'TTA argmax == view-0 argmax {100*(tta_arg == v0).mean():.1f}%; == original label {100*(tta_arg == orig_arg).mean():.1f}%; '
          f'view-0 == original {100*(v0 == orig_arg).mean():.1f}%')
    for lo, hi in ((0, .3), (.3, .5), (.5, 1.01)):
        m = (orig_top >= lo) & (orig_top < hi)
        print(f'  original top-share [{lo},{hi}): rows {100*m.mean():.1f}%  TTA==orig {100*(tta_arg[m]==orig_arg[m]).mean():.1f}%  view0==orig {100*(v0[m]==orig_arg[m]).mean():.1f}%')


if __name__ == '__main__':
    main()
