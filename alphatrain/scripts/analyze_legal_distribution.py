"""Analyze a checkpoint's legal-renormalized policy distribution.

For a given checkpoint and a set of states, computes per-state:
  - top1_legal_prob_renorm:  max(prior) / sum(top-30 legal priors)
  - top2_legal_prob_renorm:  same for rank 2
  - top1_top2_gap:           top1 - top2 (decisiveness)
  - legal_entropy:           -Σ p_renorm log p_renorm over legal top-30
  - n_legal:                 number of legal moves found in top-30

The script accepts states from either:
  - phase1_oracle_path_b.pt (oracle anchors)
  - b_selfplay_*.pt         (self-play trajectory snapshots)
  - b_anchors.pt            (sampled anchors)

Used to track whether `--target-temperature` sharpening produces a
meaningfully more decisive policy. Pre-sharpening B_ep12 should have
top1_renorm ~ 0.04 (near-uniform over top-30). Successful sharpening
should push it to 0.10+ with reduced entropy.

Usage:
    python -m alphatrain.scripts.analyze_legal_distribution \\
        --checkpoint alphatrain/data/b_smoke_epoch_12.pt \\
        --state-source alphatrain/data/phase1_oracle_path_b.pt \\
        --n-samples 2000 --device mps
"""

from __future__ import annotations

import argparse
import os
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


def states_from_source(path, n_samples, rng_seed=2026):
    """Pull a uniform sample of states from any of the standard source
    formats. Returns list of dicts with (board, next_balls, num_next).
    """
    raw = torch.load(path, weights_only=False)
    rng = np.random.default_rng(rng_seed)

    # Detect format by content
    if 'results' in raw:
        # phase1_oracle_path_b.pt (combine output) — anchor list
        items = raw['results']
        sel = rng.choice(len(items),
                          size=min(n_samples, len(items)),
                          replace=False)
        return [
            {
                'board': np.asarray(items[i]['anchor_board'],
                                      dtype=np.int8),
                'next_balls': items[i]['anchor_next_balls'],
                'num_next': int(items[i]['anchor_n_next']),
            } for i in sel
        ]
    if 'anchors' in raw:
        # sample_b_anchors.py or build_path_b_tensor.py output
        items = raw['anchors']
        # Could be list of dicts (sample_b_anchors) or tensor schema
        # (build_path_b_tensor). Detect:
        if isinstance(items, list) and items and isinstance(items[0], dict):
            sel = rng.choice(len(items),
                              size=min(n_samples, len(items)),
                              replace=False)
            return [
                {
                    'board': np.asarray(items[i]['board'], dtype=np.int8),
                    'next_balls': items[i]['next_balls'],
                    'num_next': int(items[i]['num_next']),
                } for i in sel
            ]
        else:
            raise ValueError(f"Unknown 'anchors' format in {path}")
    if 'games' in raw:
        # gen_b_selfplay output — sample one state per game
        games = raw['games']
        out = []
        for g in games:
            T = int(g['boards'].shape[0])
            if T == 0:
                continue
            i = int(rng.integers(0, T))
            nn = int(g['n_next'][i])
            next_balls = [
                ((int(g['next_pos'][i, k, 0]),
                  int(g['next_pos'][i, k, 1])),
                 int(g['next_col'][i, k]))
                for k in range(nn)
            ]
            out.append({
                'board': np.asarray(g['boards'][i], dtype=np.int8),
                'next_balls': next_balls,
                'num_next': nn,
            })
        if len(out) > n_samples:
            sel = rng.choice(len(out), size=n_samples, replace=False)
            out = [out[i] for i in sel]
        return out
    # tensor schema fallback (build_path_b_tensor output)
    if 'boards' in raw:
        N = int(raw['boards'].shape[0])
        sel = rng.choice(N, size=min(n_samples, N), replace=False)
        boards = raw['boards'].numpy()
        next_pos = raw['next_pos'].numpy()
        next_col = raw['next_col'].numpy()
        n_next = raw['n_next'].numpy()
        out = []
        for i in sel:
            nn = int(n_next[i])
            next_balls = [
                ((int(next_pos[i, k, 0]), int(next_pos[i, k, 1])),
                 int(next_col[i, k]))
                for k in range(nn)
            ]
            out.append({
                'board': boards[i].astype(np.int8),
                'next_balls': next_balls,
                'num_next': nn,
            })
        return out
    raise ValueError(f"Could not detect format of {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--state-source', required=True,
                   help='Path to .pt of any standard format (oracle '
                        'anchors, sampled anchors, self-play, tensor).')
    p.add_argument('--n-samples', type=int, default=2000)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--device', default=None)
    p.add_argument('--legal-top-k', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--rng-seed', type=int, default=2026)
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

    print(f"\nLoading {args.checkpoint}...", flush=True)
    model, epoch = load_b_model(args.checkpoint, device,
                                  args.num_blocks, args.channels)
    val_loss = None
    try:
        ck = torch.load(args.checkpoint, map_location='cpu',
                          weights_only=False)
        val_loss = ck.get('val_loss', None)
    except Exception:
        pass
    print(f"  epoch={epoch}  val_loss="
          f"{f'{val_loss:.4f}' if val_loss is not None else 'n/a'}",
          flush=True)

    print(f"\nLoading states from {args.state_source}...", flush=True)
    states = states_from_source(args.state_source, args.n_samples,
                                  args.rng_seed)
    print(f"  {len(states)} states sampled", flush=True)

    @torch.no_grad()
    def _forward(obs_np):
        ob = torch.from_numpy(obs_np).to(device)
        out = model(ob)
        if isinstance(out, tuple):
            out = out[0]
        return torch.softmax(out.float(), dim=-1).cpu().numpy()

    print(f"\nForwarding (batch={args.batch_size})...", flush=True)
    t0 = time.time()

    top1_renorm = np.zeros(len(states), dtype=np.float32)
    top2_renorm = np.zeros(len(states), dtype=np.float32)
    gap = np.zeros(len(states), dtype=np.float32)
    legal_entropy = np.zeros(len(states), dtype=np.float32)
    n_legal = np.zeros(len(states), dtype=np.int32)
    # Also un-renormalized top1 (for comparison)
    top1_raw = np.zeros(len(states), dtype=np.float32)

    for start in range(0, len(states), args.batch_size):
        sub = states[start:start + args.batch_size]
        obs_batch = np.stack([
            build_observation(
                rec['board'],
                np.array([nb[0][0] for nb in rec['next_balls']],
                          dtype=np.int64),
                np.array([nb[0][1] for nb in rec['next_balls']],
                          dtype=np.int64),
                np.array([nb[1] for nb in rec['next_balls']],
                          dtype=np.int64),
                int(rec['num_next']),
            )
            for rec in sub
        ])
        pol = _forward(obs_batch)
        for k, rec in enumerate(sub):
            priors = _get_legal_priors_flat(rec['board'], pol[k],
                                              args.legal_top_k)
            i = start + k
            if not priors:
                n_legal[i] = 0
                top1_renorm[i] = 1.0
                continue
            vals = sorted(priors.values(), reverse=True)
            total = sum(vals)
            n_legal[i] = len(vals)
            if total > 0:
                renorm = [v / total for v in vals]
                top1_renorm[i] = renorm[0]
                top1_raw[i] = vals[0]
                if len(renorm) >= 2:
                    top2_renorm[i] = renorm[1]
                gap[i] = renorm[0] - top2_renorm[i]
                legal_entropy[i] = -float(np.sum([
                    r * np.log(r) for r in renorm if r > 0
                ]))
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # ── Aggregate ──
    print(f"\n=== Legal-renormalized policy distribution stats ===")
    print(f"({len(states)} states sampled from {args.state_source})", flush=True)
    print()
    print(f"  n_legal:   "
          f"mean {n_legal.mean():.1f}  "
          f"P10 {np.percentile(n_legal, 10):.0f}  "
          f"P50 {np.percentile(n_legal, 50):.0f}  "
          f"P90 {np.percentile(n_legal, 90):.0f}  "
          f"max {n_legal.max()}", flush=True)
    print(f"  top1_renorm: "
          f"mean {top1_renorm.mean():.3f}  "
          f"P10 {np.percentile(top1_renorm, 10):.3f}  "
          f"P50 {np.percentile(top1_renorm, 50):.3f}  "
          f"P90 {np.percentile(top1_renorm, 90):.3f}  "
          f"P99 {np.percentile(top1_renorm, 99):.3f}", flush=True)
    print(f"  top1_raw:    "
          f"mean {top1_raw.mean():.3f}  "
          f"P10 {np.percentile(top1_raw, 10):.3f}  "
          f"P50 {np.percentile(top1_raw, 50):.3f}  "
          f"P90 {np.percentile(top1_raw, 90):.3f}  "
          f"P99 {np.percentile(top1_raw, 99):.3f}", flush=True)
    print(f"  top2_renorm: "
          f"mean {top2_renorm.mean():.3f}  "
          f"P50 {np.percentile(top2_renorm, 50):.3f}", flush=True)
    print(f"  top1-top2 gap: "
          f"mean {gap.mean():.3f}  "
          f"P10 {np.percentile(gap, 10):.3f}  "
          f"P50 {np.percentile(gap, 50):.3f}  "
          f"P90 {np.percentile(gap, 90):.3f}", flush=True)
    print(f"  legal entropy: "
          f"mean {legal_entropy.mean():.3f}  "
          f"P50 {np.percentile(legal_entropy, 50):.3f}  "
          f"max {legal_entropy.max():.3f}", flush=True)

    # Decisiveness bucketing
    print(f"\n  Decisiveness (top1_renorm bucket):", flush=True)
    for lo, hi in ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20),
                    (0.20, 0.40), (0.40, 1.01)):
        n = ((top1_renorm >= lo) & (top1_renorm < hi)).sum()
        print(f"    [{lo:.2f}, {hi:.2f}): {n} "
              f"({100*n/len(top1_renorm):.1f}%)", flush=True)

    # Uniform-over-K reference
    print(f"\n  Reference: uniform-over-30 = 0.033, "
          f"uniform-over-{int(n_legal.mean())} = "
          f"{1/max(n_legal.mean(),1):.3f}", flush=True)
    if top1_renorm.mean() < 0.10:
        print(f"\n  ⚠  Policy is barely above uniform on top-K — "
              f"under-committed. Sharpening expected to help.",
              flush=True)
    elif top1_renorm.mean() > 0.50:
        print(f"\n  ✓  Policy is decisive (top1_renorm mean > 0.5).",
              flush=True)


if __name__ == '__main__':
    main()
