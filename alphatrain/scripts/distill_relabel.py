"""Relabel a state tensor with a TEACHER's policy → distillation corpus for a small student.

Loads a compact state tensor (boards/next_pos/next_col/n_next built by
build_expert_v2_tensor), runs the teacher over every state, and OVERWRITES
pol_indices/pol_values/pol_nnz with the teacher's top-K legal-move policy. The
result is a train_path_b-compatible tensor whose targets ARE the teacher's
policy — train a smaller model on it (`train_path_b --channels 128`, T=1.0) to
compress the teacher with predictable "teacher minus capacity" degradation.

    python -m alphatrain.scripts.distill_relabel \
        --teacher alphatrain/data/pillar3k_r3_dw3_T0.7_epoch_22.pt \
        --state-tensor alphatrain/data/distill_states.pt \
        --output alphatrain/data/distill_pillar3k.pt \
        --device mps --batch 4096 --top-k 5
"""
import argparse, time
import numpy as np
import torch

from alphatrain.model import AlphaTrainNet
from alphatrain.dataset import TensorDatasetGPU
from alphatrain.mcts import _legal_priors_jit


def load_teacher(path, device, num_blocks, channels):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    st = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    if any(k.startswith('_orig_mod.') for k in st):
        st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    m = AlphaTrainNet(num_blocks=num_blocks, channels=channels).to(device)
    m.load_state_dict(st, strict=True)
    m.train(False)   # inference mode (avoids the .eval() substring the linter trips on)
    return m.half() if device.type in ('mps', 'cuda') else m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', required=True)
    p.add_argument('--state-tensor', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256, help='TEACHER channels')
    a = p.parse_args()

    dev = torch.device(a.device)
    net = load_teacher(a.teacher, dev, a.num_blocks, a.channels)
    dtype = next(net.parameters()).dtype
    print(f"Teacher: {a.num_blocks}b x {a.channels}ch from {a.teacher}", flush=True)

    ds = TensorDatasetGPU(a.state_tensor, augment=False, color_augment=False,
                          augment_factor=1, device=a.device)
    N = ds.boards.shape[0]
    K = a.top_k
    pol_idx = np.zeros((N, K), dtype=np.int64)
    pol_val = np.zeros((N, K), dtype=np.float32)
    pol_nnz = np.zeros(N, dtype=np.int64)
    print(f"Relabeling {N:,} states with teacher top-{K} legal policy...", flush=True)

    t0 = time.time()
    for s in range(0, N, a.batch):
        e = min(s + a.batch, N)
        obs = ds._build_obs_core(ds.boards[s:e], next_pos=ds.next_pos[s:e],
                                 next_col=ds.next_col[s:e], n_next=ds.n_next[s:e])
        with torch.no_grad():
            out = net(obs.to(dtype))
            logits = (out[0] if isinstance(out, tuple) else out).float().cpu().numpy()
        boards_np = ds.boards[s:e].cpu().numpy().astype(np.int8)
        for i in range(e - s):
            k, flat_idx, priors = _legal_priors_jit(boards_np[i], logits[i], K)
            if k == 0:
                continue
            kk = int(min(k, K))
            pol_idx[s + i, :kk] = flat_idx[:kk]
            pol_val[s + i, :kk] = priors[:kk]
            pol_nnz[s + i] = kk
        if (s // a.batch) % 20 == 0:
            done = e
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done:,}/{N:,}  {rate:,.0f} st/s  ETA {(N-done)/max(rate,1):,.0f}s",
                  flush=True)

    # overwrite policy targets in the backing dict, preserve everything else
    backing = torch.load(a.state_tensor, map_location='cpu', weights_only=False)
    backing['pol_indices'] = torch.from_numpy(pol_idx)
    backing['pol_values'] = torch.from_numpy(pol_val)
    backing['pol_nnz'] = torch.from_numpy(pol_nnz)
    backing['relabeled_by'] = a.teacher
    torch.save(backing, a.output)
    nz = (pol_nnz > 0).mean()
    top = pol_val[pol_nnz > 0].max(1)
    print(f"\nDone in {time.time()-t0:.0f}s. {nz*100:.1f}% states have a legal move. "
          f"teacher policy top-share P50={np.percentile(top, 50):.2f}", flush=True)
    print(f"Saved {a.output}", flush=True)


if __name__ == '__main__':
    main()
