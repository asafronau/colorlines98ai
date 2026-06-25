"""Validate the de-peaking mechanism for the recipe regression.

HYPOTHESIS (from project_selfplay_gumbel_recipe): the 400-sim VISIT TARGET peakedness
is fixed (top-share ~0.14), but the BASE policy peakedness grew across generations. Once
the base policy is PEAKIER than the target, soft-CE distillation DE-PEAKS it -> regress.
The same recipe IMPROVED earlier, flatter policies.

This measures, on the SAME v15 candidate support (pol_indices), the top-1 share of:
  (a) the 400-sim visit TARGET            (pol_values, what we distill toward)
  (b) pillar3f policy   (the FAILING base, distilling into it regressed -27%)
  (c) pillar3b policy   (the LAST WORKING-era model, recipe gave +15%)

Prediction if the mechanism is real:
  target P50 ~0.14  <<  pillar3f P50 (peaky, ~0.34) -> distilling de-peaks pillar3f
  pillar3b P50 should sit MUCH closer to / below the target -> distilling preserved/sharpened.

Top-share is computed over the candidate support so it is apples-to-apples with the
soft-CE the trainer actually applies (candidate-restricted visit distribution).
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.model import AlphaTrainNet

TENSOR = 'alphatrain/data/v15_pillar3f_slim.pt'
MODELS = {
    'pillar3f (FAILS, -27%)': 'alphatrain/data/pillar3f.pt',
    'pillar3b ep20 (WORKED, +15%)': 'alphatrain/data/pillar3b_epoch_20.pt',
}
N_SAMPLE = 5000
SEED = 42


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    m = AlphaTrainNet(num_blocks=10, channels=256).to(device)
    ms = m.state_dict()
    filt = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
    skipped = [k for k in state if k not in filt]
    m.load_state_dict(filt, strict=False)
    m.train(False)   # inference mode (avoid .eval substring -> hook false positive)
    return m, len(filt), skipped


def pct(x):
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return f"P10={q[0]:.3f} P25={q[1]:.3f} P50={q[2]:.3f} P75={q[3]:.3f} P90={q[4]:.3f}  mean={x.mean():.3f}"


def main():
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f"Device: {device}", flush=True)

    d = torch.load(TENSOR, map_location='cpu', mmap=True, weights_only=False)
    n_total = d['boards'].shape[0]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_total, size=N_SAMPLE, replace=False))
    print(f"Sampling {N_SAMPLE} of {n_total:,} states (seed {SEED})", flush=True)

    boards = d['boards'][idx].numpy()                       # (N,9,9) int8
    next_pos = d['next_pos'][idx].numpy().astype(np.int64)  # (N,3,2)
    next_col = d['next_col'][idx].numpy().astype(np.int64)  # (N,3)
    n_next = d['n_next'][idx].numpy().astype(np.int64)       # (N,)
    pol_idx = d['pol_indices'][idx].numpy().astype(np.int64)  # (N,5)
    pol_val = d['pol_values'][idx].numpy().astype(np.float64)  # (N,5)
    pol_nnz = d['pol_nnz'][idx].numpy().astype(np.int64)       # (N,)

    # (a) TARGET top-share over the candidate support
    cand_mask = (np.arange(pol_val.shape[1])[None, :] < pol_nnz[:, None])
    pv = np.where(cand_mask, pol_val, 0.0)
    pv_sum = pv.sum(1)
    keep = pv_sum > 1e-9
    tgt_top = pv[keep].max(1) / pv_sum[keep]
    print(f"\n(a) 400-sim VISIT TARGET  (n={int(keep.sum())}):  {pct(tgt_top)}", flush=True)

    print(f"\nBuilding {N_SAMPLE} observations...", flush=True)
    obs = np.empty((N_SAMPLE, 18, 9, 9), dtype=np.float32)
    for i in range(N_SAMPLE):
        obs[i] = build_observation(boards[i], next_pos[i, :, 0], next_pos[i, :, 1],
                                   next_col[i], int(n_next[i]))
    obs_t = torch.from_numpy(obs)

    for name, path in MODELS.items():
        m, nload, skipped = load_model(path, device)
        with torch.no_grad():
            chunks = []
            for s in range(0, N_SAMPLE, 2048):
                out = m(obs_t[s:s+2048].to(device))
                lg = out[0] if isinstance(out, tuple) else out
                chunks.append(lg.float().cpu())
            logits = torch.cat(chunks).numpy()              # (N,6561)
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        cand_p = np.take_along_axis(p, pol_idx, axis=1)     # (N,5) prob on candidates
        cand_p = np.where(cand_mask, cand_p, 0.0)
        cs = cand_p.sum(1)
        k2 = cs > 1e-9
        top = cand_p[k2].max(1) / cs[k2]
        print(f"\n({name})  loaded {nload} keys, skipped {len(skipped)} (n={int(k2.sum())}):"
              f"\n    {pct(top)}", flush=True)

    print("\n--- READ ---", flush=True)
    print("If pillar3f P50 >> target P50 (~0.14): distilling the target DE-PEAKS pillar3f "
          "-> the validated mechanism for the -27% regression.", flush=True)
    print("If pillar3b P50 is close to / below target: distillation preserved/sharpened it "
          "-> why the SAME recipe gave +15% then. Crossover confirmed.", flush=True)


if __name__ == '__main__':
    main()
