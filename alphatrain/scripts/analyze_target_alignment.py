"""Compare a trained policy's output to its V12 training targets on the
SAME action set. Tests whether the model is genuinely under-committed
or just appearing flat because we previously compared mismatched sets.

For each sampled V12 state:
  - target_actions    : indices where pol_values > 0 (top-5)
  - target_top1_action: highest pol_value action
  - target_top1_prob  : the target's top-1 probability (≈ 0.26 on V12)
  - model legal softmax = softmax(logits) renormalized over legal moves

Per-state metrics:
  - model_prob_at_target_top1  : ↑ means B IS committing to target winner
  - model_mass_on_target_set   : ↑ means B concentrates probability on
                                  the stored targets (not other legals)
  - rank_target_top1_in_legal  : 1 = B's argmax matches target_top1
  - target_set_CE              : cross-entropy on the target distribution
                                  using the model's renormalized-on-
                                  target-set probabilities

Usage:
    python -m alphatrain.scripts.analyze_target_alignment \\
        --checkpoint alphatrain/data/b_smoke_epoch_12.pt \\
        --tensor-file alphatrain/data/v12_pillar2z.pt \\
        --n-samples 5000 --device mps
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from alphatrain.mcts import _get_legal_priors_flat
from alphatrain.model import AlphaTrainNet
from alphatrain.observation import build_observation


def load_b_model(checkpoint_path, device, num_blocks=10, channels=256):
    ckpt = torch.load(checkpoint_path, map_location=device,
                       weights_only=False)
    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model = AlphaTrainNet(num_blocks=num_blocks,
                          channels=channels).to(device)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                 if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.train(False)
    return model, ckpt.get('epoch', '?')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tensor-file', required=True,
                   help='V12 .pt with boards, next_pos, next_col, '
                        'n_next, pol_indices, pol_values, pol_nnz')
    p.add_argument('--n-samples', type=int, default=5000)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--device', default='mps')
    p.add_argument('--legal-top-k', type=int, default=60,
                   help='How many top legal moves to consider for the '
                        'renormalized softmax (60 to capture full legal set '
                        'reliably).')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # ── Load model ──
    print(f"\nLoading {args.checkpoint}...", flush=True)
    model, epoch = load_b_model(args.checkpoint, device,
                                  args.num_blocks, args.channels)
    print(f"  epoch={epoch}", flush=True)

    # ── Load V12 tensor ──
    print(f"\nLoading {args.tensor_file}...", flush=True)
    t0 = time.time()
    data = torch.load(args.tensor_file, weights_only=False)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    boards = data['boards']
    next_pos = data['next_pos']
    next_col = data['next_col']
    n_next = data['n_next']
    pol_indices = data['pol_indices']
    pol_values = data['pol_values']
    pol_nnz = data.get('pol_nnz', None)
    if isinstance(boards, torch.Tensor):
        boards = boards.numpy()
        next_pos = next_pos.numpy()
        next_col = next_col.numpy()
        n_next = n_next.numpy()
        pol_indices = pol_indices.numpy()
        pol_values = pol_values.numpy()
        if pol_nnz is not None:
            pol_nnz = pol_nnz.numpy()

    N = boards.shape[0]
    K_target = pol_indices.shape[1]
    print(f"  {N:,} states, top-{K_target} sparse targets", flush=True)

    sel = rng.choice(N, size=min(args.n_samples, N), replace=False)
    print(f"  sampling {len(sel)} states", flush=True)

    @torch.no_grad()
    def _forward(obs_np):
        ob = torch.from_numpy(obs_np).to(device)
        out = model(ob)
        if isinstance(out, tuple):
            out = out[0]
        return torch.softmax(out.float(), dim=-1).cpu().numpy()

    # ── Compute metrics ──
    print(f"\nForwarding {len(sel)} states (batch={args.batch_size})...",
          flush=True)
    t0 = time.time()

    # Per-state arrays
    target_top1_prob = np.zeros(len(sel), dtype=np.float32)
    model_prob_at_target_top1 = np.zeros(len(sel), dtype=np.float32)
    model_mass_on_target_set = np.zeros(len(sel), dtype=np.float32)
    rank_target_top1 = np.zeros(len(sel), dtype=np.int32)
    target_set_ce = np.zeros(len(sel), dtype=np.float32)
    target_top1_in_legal = np.zeros(len(sel), dtype=bool)

    for start in range(0, len(sel), args.batch_size):
        sub_idx = sel[start:start + args.batch_size]
        # Build obs batch
        obs_batch = np.empty((len(sub_idx), 18, 9, 9), dtype=np.float32)
        for k, i in enumerate(sub_idx):
            nn_i = int(n_next[i])
            obs_batch[k] = build_observation(
                boards[i],
                next_pos[i, :nn_i, 0].astype(np.int64),
                next_pos[i, :nn_i, 1].astype(np.int64),
                next_col[i, :nn_i].astype(np.int64),
                nn_i,
            )
        pol = _forward(obs_batch)

        for k, i in enumerate(sub_idx):
            out_idx = start + k
            # Target distribution
            nnz = (int(pol_nnz[i]) if pol_nnz is not None
                    else int((pol_values[i] > 0).sum()))
            if nnz < 1:
                continue
            t_actions = pol_indices[i, :nnz].astype(np.int64)
            t_probs = pol_values[i, :nnz].astype(np.float64)
            t_probs = t_probs / max(t_probs.sum(), 1e-12)  # safety
            t_top1_idx = int(np.argmax(t_probs))
            t_top1_action = int(t_actions[t_top1_idx])
            target_top1_prob[out_idx] = float(t_probs[t_top1_idx])

            # Legal priors from model softmax over 6561
            priors = _get_legal_priors_flat(boards[i], pol[k],
                                              args.legal_top_k)
            if not priors:
                continue
            total = sum(priors.values())
            if total <= 0:
                continue
            # Renormalize over legal
            legal_renorm = {a: v / total for a, v in priors.items()}

            # Model's prob at target_top1
            target_top1_in_legal[out_idx] = (t_top1_action in legal_renorm)
            model_prob_at_target_top1[out_idx] = (
                legal_renorm.get(t_top1_action, 0.0))
            # Mass on target set
            mass = sum(legal_renorm.get(int(a), 0.0) for a in t_actions)
            model_mass_on_target_set[out_idx] = mass
            # Rank of target_top1 among legal
            sorted_legal = sorted(legal_renorm.items(), key=lambda kv: -kv[1])
            r = 1
            for action, _ in sorted_legal:
                if action == t_top1_action:
                    break
                r += 1
            else:
                r = len(sorted_legal) + 1  # not in legal at all
            rank_target_top1[out_idx] = r

            # CE on target set: cross-entropy of target distribution wrt
            # model probabilities renormalized over the target set.
            target_set_mass = sum(legal_renorm.get(int(a), 0.0)
                                   for a in t_actions)
            if target_set_mass > 0:
                ce = 0.0
                for a, tp in zip(t_actions, t_probs):
                    mp = legal_renorm.get(int(a), 0.0)
                    mp_renorm = mp / target_set_mass
                    if mp_renorm > 1e-12 and tp > 0:
                        ce -= float(tp) * float(np.log(mp_renorm))
                target_set_ce[out_idx] = ce
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # ── Report ──
    valid = target_top1_prob > 0
    n_valid = int(valid.sum())
    print(f"\n=== Target-set alignment (over {n_valid} states) ===")
    print(f"  target top1_prob:              "
          f"mean {target_top1_prob[valid].mean():.3f}  "
          f"P50 {np.median(target_top1_prob[valid]):.3f}  "
          f"P90 {np.percentile(target_top1_prob[valid], 90):.3f}")
    print(f"  model prob @ target_top1:      "
          f"mean {model_prob_at_target_top1[valid].mean():.3f}  "
          f"P50 {np.median(model_prob_at_target_top1[valid]):.3f}  "
          f"P10 {np.percentile(model_prob_at_target_top1[valid], 10):.3f}  "
          f"P90 {np.percentile(model_prob_at_target_top1[valid], 90):.3f}")
    print(f"  model mass on target set:      "
          f"mean {model_mass_on_target_set[valid].mean():.3f}  "
          f"P50 {np.median(model_mass_on_target_set[valid]):.3f}  "
          f"P10 {np.percentile(model_mass_on_target_set[valid], 10):.3f}  "
          f"P90 {np.percentile(model_mass_on_target_set[valid], 90):.3f}")
    print(f"  target_top1 in legal set:      "
          f"{100*target_top1_in_legal[valid].mean():.1f}%")
    print(f"  rank of target_top1 in legal:  "
          f"mean {rank_target_top1[valid].mean():.1f}  "
          f"P50 {np.median(rank_target_top1[valid]):.1f}  "
          f"P90 {np.percentile(rank_target_top1[valid], 90):.1f}  "
          f"rank=1 (B's argmax = target): "
          f"{100*(rank_target_top1[valid] == 1).mean():.1f}%")
    print(f"  target-set CE:                 "
          f"mean {target_set_ce[valid].mean():.4f}  "
          f"P50 {np.median(target_set_ce[valid]):.4f}  "
          f"min {target_set_ce[valid].min():.4f}")

    print(f"\nInterpretation key:")
    print(f"  If model prob @ target_top1 ≈ target top1_prob:")
    print(f"      → B faithfully fits training targets on target set.")
    print(f"      → 'Flat over legal' was a measurement artifact.")
    print(f"  If model prob @ target_top1 << target top1_prob:")
    print(f"      → B is under-committed; sharpening should help.")
    print(f"  If model mass on target set < 0.5:")
    print(f"      → B redistributes mass to non-target legal moves;")
    print(f"        sharpening targets should reclaim that mass.")


if __name__ == '__main__':
    main()
