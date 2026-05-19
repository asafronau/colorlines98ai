"""Post-hoc oracle metrics for any Path B checkpoint.

Loads an epoch checkpoint + the Path B oracle tensor, runs the model on
the oracle val anchors, and reports:

  - top1 agreement per Δcap bucket
  - mean logit gap (oracle_best − policy_top1)
  - mean P(oracle_best | softmax over top-6) per bucket
  - weighted KL per bucket

Buckets are the same ones the reliability ramp uses:
  (0.00, 0.05]  noise floor
  (0.05, 0.10]  weak
  (0.10, 0.15]  medium
  (0.15, 0.25]  strong
  (0.25, 1.00]  dominant

Usage:
    python -m alphatrain.scripts.analyze_path_b_checkpoint \\
        --checkpoint /path/to/path_b_C_smoke_epoch_5.pt \\
        --oracle-tensor alphatrain/data/phase1_oracle_path_b.pt \\
        --val-only --device cuda
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from alphatrain.model import AlphaTrainNet
from alphatrain.train_path_b import (
    NEG_INF,
    OracleDataset,
    reliability_weight,
)


# Match the reliability ramp boundaries
BUCKETS = [
    ('noise',    0.00, 0.05),
    ('weak',     0.05, 0.10),
    ('medium',   0.10, 0.15),
    ('strong',   0.15, 0.25),
    ('dominant', 0.25, 1.001),
]


def load_model(checkpoint_path, num_blocks, channels, device):
    """Load model weights from a Path B / train.py checkpoint."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ck['model']
    # Strip torch.compile prefix if present
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model = AlphaTrainNet(num_blocks=num_blocks, channels=channels).to(device)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                 if k in model_state and v.shape == model_state[k].shape}
    skipped = [k for k in state if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    model.train(False)
    epoch = ck.get('epoch', '?')
    val_loss = ck.get('val_loss', float('nan'))
    return model, epoch, val_loss, skipped


@torch.no_grad()
def gather_metrics(model, ods, indices, device, beta=10.0,
                    noise_floor=0.05, scale=0.20, batch_size=4096):
    """Run forward on `indices` of the oracle dataset.

    Returns per-anchor arrays:
        delta_cap, gap_top6, p_oracle_top6, kl, top1_agree,
        entropy_full, p_top1_full, oracle_best_global_rank
    """
    n = int(indices.numel())
    out_dcap = torch.empty(n, dtype=torch.float32)
    out_gap = torch.empty(n, dtype=torch.float32)
    out_p = torch.empty(n, dtype=torch.float32)
    out_kl = torch.empty(n, dtype=torch.float32)
    out_agree = torch.empty(n, dtype=torch.bool)
    out_ent_full = torch.empty(n, dtype=torch.float32)
    out_p_top1_full = torch.empty(n, dtype=torch.float32)
    out_global_rank = torch.empty(n, dtype=torch.int32)

    cursor = 0
    for start in range(0, n, batch_size):
        sel = indices[start:start + batch_size]
        obs = ods.obs[sel]
        actions = ods.actions[sel]
        cap_rates = ods.cap_rates[sel]
        n_moves = ods.n_moves[sel]
        delta_cap = ods.delta_cap[sel]
        B = obs.shape[0]

        out = model(obs)
        logits = (out[0] if isinstance(out, tuple) else out).float()

        # ── Full-distribution metrics (over all 6561 logits) ──
        full_log_p = F.log_softmax(logits, dim=-1)
        full_p = full_log_p.exp()
        entropy_full = -(full_p * full_log_p).sum(dim=-1)
        p_top1_full = full_p.max(dim=-1).values

        # ── Top-6 conditional metrics ──
        K = actions.shape[1]
        arange_k = torch.arange(K, device=device)
        mask = arange_k.unsqueeze(0) < n_moves.to(torch.long).unsqueeze(1)

        safe_actions = actions.clamp(min=0)
        gathered = logits.gather(1, safe_actions)
        cand_logits = torch.where(
            mask, gathered, torch.full_like(gathered, NEG_INF))

        score = torch.where(mask, beta * cap_rates,
                              torch.full_like(cap_rates, NEG_INF))
        target = F.softmax(score, dim=-1)
        oracle_best_slot = score.argmax(dim=-1)
        policy_top1_slot = cand_logits.argmax(dim=-1)

        log_pred = F.log_softmax(cand_logits, dim=-1)
        pred = log_pred.exp()
        log_target = torch.log(target.clamp(min=1e-30))
        kl = target * (log_target - log_pred)
        kl = torch.where(mask, kl, torch.zeros_like(kl))
        kl_per = kl.sum(dim=-1)

        rng = torch.arange(B, device=device)
        logit_oracle = cand_logits[rng, oracle_best_slot]
        logit_top1 = cand_logits[rng, policy_top1_slot]
        p_oracle = pred[rng, oracle_best_slot]
        gap = logit_oracle - logit_top1

        # ── Oracle-best global rank in full 6561 distribution ──
        # For each anchor, the oracle's chosen action_int (one of the 6
        # candidates); count actions across all 6561 with higher logit.
        oracle_action = safe_actions[rng, oracle_best_slot]   # (B,)
        oracle_logit = logits[rng, oracle_action]              # (B,)
        # Rank = 1 + count(logits > oracle_logit) over the full row.
        higher_mask = logits > oracle_logit.unsqueeze(-1)
        global_rank = higher_mask.sum(dim=-1) + 1              # (B,)

        out_dcap[cursor:cursor + B] = delta_cap.cpu()
        out_gap[cursor:cursor + B] = gap.cpu()
        out_p[cursor:cursor + B] = p_oracle.cpu()
        out_kl[cursor:cursor + B] = kl_per.cpu()
        out_agree[cursor:cursor + B] = (
            oracle_best_slot == policy_top1_slot).cpu()
        out_ent_full[cursor:cursor + B] = entropy_full.cpu()
        out_p_top1_full[cursor:cursor + B] = p_top1_full.cpu()
        out_global_rank[cursor:cursor + B] = global_rank.to(torch.int32).cpu()
        cursor += B

    return (out_dcap, out_gap, out_p, out_kl, out_agree,
             out_ent_full, out_p_top1_full, out_global_rank)


def report(dcap, gap, p, kl, agree, noise_floor, scale):
    """Print a per-bucket summary."""
    w = reliability_weight(dcap, noise_floor, scale)

    print(f"\n{'bucket':<10} {'n':>6} {'logit_gap':>11} {'P(oracle)':>11} "
          f"{'KL':>8} {'KL_w':>8} {'top1':>6}")
    print('-' * 64)
    for name, lo, hi in BUCKETS:
        m = (dcap >= lo) & (dcap < hi)
        n = int(m.sum().item())
        if n == 0:
            print(f"{name:<10} {n:>6}  (no anchors)")
            continue
        bucket_gap = gap[m].mean().item()
        bucket_p = p[m].mean().item()
        bucket_kl = kl[m].mean().item()
        bucket_w = w[m]
        if bucket_w.sum().item() > 0:
            bucket_kl_w = (
                (bucket_w * kl[m]).sum() / bucket_w.sum()).item()
        else:
            bucket_kl_w = 0.0
        bucket_top1 = agree[m].float().mean().item()
        print(f"{name:<10} {n:>6} {bucket_gap:>+11.3f} "
              f"{bucket_p:>11.3f} {bucket_kl:>8.4f} "
              f"{bucket_kl_w:>8.4f} {bucket_top1:>6.3f}")

    n_total = int(dcap.numel())
    print('-' * 64)
    print(f"{'TOTAL':<10} {n_total:>6} "
          f"{gap.mean().item():>+11.3f} "
          f"{p.mean().item():>11.3f} "
          f"{kl.mean().item():>8.4f} "
          f"{((w * kl).sum() / w.sum().clamp(min=1.0)).item():>8.4f} "
          f"{agree.float().mean().item():>6.3f}")

    print("\nlogit_gap = mean(policy_logit[oracle_best] − policy_logit[policy_top1])")
    print("  positive = model already prefers oracle pick (agreement on argmax)")
    print("  negative = model disagrees; magnitude = how strongly")
    print("P(oracle) = mean prob mass on oracle_best under softmax over top-6")
    print(f"KL_w = reliability-weighted KL (noise_floor={noise_floor}, "
          f"scale={scale})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--oracle-tensor', required=True)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--device', default=None,
                   help='cuda / mps / cpu (auto-detect if omitted)')
    p.add_argument('--val-only', action='store_true',
                   help='Restrict to the same val split the trainer used '
                        '(oracle-seed 2026, val_frac 0.05).')
    p.add_argument('--oracle-seed', type=int, default=2026)
    p.add_argument('--oracle-val-frac', type=float, default=0.05)
    p.add_argument('--beta', type=float, default=10.0)
    p.add_argument('--noise-floor', type=float, default=0.05)
    p.add_argument('--scale', type=float, default=0.20)
    p.add_argument('--batch-size', type=int, default=4096)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    print(f"\nLoading checkpoint {args.checkpoint}...", flush=True)
    model, epoch, val_loss, skipped = load_model(
        args.checkpoint, args.num_blocks, args.channels, device)
    print(f"  epoch={epoch}  val_loss={val_loss:.4f}", flush=True)
    if skipped:
        print(f"  skipped {len(skipped)} keys (shape/name mismatch)",
              flush=True)

    print(f"\nLoading oracle tensor {args.oracle_tensor}...", flush=True)
    ods = OracleDataset(args.oracle_tensor, device=device)
    if args.val_only:
        _, val_idx = ods.split(val_frac=args.oracle_val_frac,
                                  seed=args.oracle_seed)
        idx = val_idx
        scope = 'val'
    else:
        idx = torch.arange(ods.N, device=device)
        scope = 'all'
    print(f"  {scope}: {int(idx.numel())} anchors", flush=True)

    print(f"\nGathering metrics...", flush=True)
    dcap, gap, p_oracle, kl, agree = gather_metrics(
        model, ods, idx, device,
        beta=args.beta,
        noise_floor=args.noise_floor,
        scale=args.scale,
        batch_size=args.batch_size,
    )

    print(f"\n=== Oracle metrics ({scope} set, "
          f"n={int(dcap.numel())}, β={args.beta}) ===")
    report(dcap, gap, p_oracle, kl, agree,
            args.noise_floor, args.scale)


if __name__ == '__main__':
    main()
