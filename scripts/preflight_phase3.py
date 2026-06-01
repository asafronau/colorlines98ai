"""Phase 3 preflight: baseline metrics on the counterfactual corpus.

Loads pillar3b (or the warm-start checkpoint), runs it on the filtered
~528 anchors of stationary_counterfactuals_v1.pt, and reports:
  - stored_top1_flip_rate: should be near 0 at baseline (top1 is rank 1
    by prior, so its logit should dominate)
  - all_clean_loser_margin_rate (at margin 0.25): should be near 0
  - Distribution of (logit[winner] − logit[stored_top1]): the empirical
    scale check for the margin hyperparameter

Run BEFORE Phase 3 training to confirm the target margin (0.25) is
appropriate. If the median logit-gap is wildly off ±0.25, adjust.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from alphatrain.counterfactual import (
    build_corpus, listwise_margin_loss, preflight_metrics, DEFAULT_MARGIN)
from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation


def build_obs_batch(corpus, device):
    """Build (N, 18, 9, 9) observations for all anchors, CPU-numba then move."""
    N = corpus['boards'].size(0)
    boards = corpus['boards'].cpu().numpy().astype(np.int8)
    next_pos = corpus['next_pos'].cpu().numpy().astype(np.int8)
    next_col = corpus['next_col'].cpu().numpy().astype(np.int8)
    n_next = corpus['n_next'].cpu().numpy().astype(np.int8)

    obs = np.zeros((N, 18, 9, 9), dtype=np.float32)
    for i in range(N):
        nr = next_pos[i, :, 0].astype(np.int64)
        nc = next_pos[i, :, 1].astype(np.int64)
        nco = next_col[i].astype(np.int64)
        obs[i] = build_observation(boards[i], nr, nc, nco, int(n_next[i]))
    return torch.from_numpy(obs).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--records',
                   default='alphatrain/data/stationary_counterfactuals_v1.pt')
    p.add_argument('--margin', type=float, default=DEFAULT_MARGIN)
    p.add_argument('--batch-size', type=int, default=512)
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    corpus = build_corpus(args.records, device=str(device))
    N = corpus['boards'].size(0)
    print(f"Corpus: {N} anchors, "
          f"{int(corpus['loser_mask'].sum())} clean-other pairs", flush=True)

    print("Building observations...", flush=True)
    obs = build_obs_batch(corpus, device)

    net, _ = load_model(args.model, device)
    net.train(False)

    print("Forward pass...", flush=True)
    logits_chunks = []
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            chunk = obs[i:i+args.batch_size]
            out = net(chunk)
            logits_chunks.append(out[0] if isinstance(out, tuple) else out)
    logits = torch.cat(logits_chunks, dim=0)

    # Baseline metrics
    metrics = preflight_metrics(
        logits, corpus['winner_idx'], corpus['top1_idx'],
        corpus['loser_idx'], corpus['loser_mask'], margin=args.margin)
    print("\n=== Baseline metrics (margin={}) ===".format(args.margin))
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Baseline listwise loss
    loss = listwise_margin_loss(
        logits, corpus['winner_idx'], corpus['top1_idx'],
        corpus['loser_idx'], corpus['loser_mask'], margin=args.margin)
    print(f"  baseline_listwise_loss: {float(loss):.4f}")

    # Logit-gap distributions
    win_l = logits.gather(1, corpus['winner_idx'].unsqueeze(1)).squeeze(1)
    top1_l = logits.gather(1, corpus['top1_idx'].unsqueeze(1)).squeeze(1)
    gap_top1 = (win_l - top1_l).cpu().float().numpy()

    other_l = logits.gather(1, corpus['loser_idx'])
    other_gap = (win_l.unsqueeze(1) - other_l).cpu().float().numpy()
    mask = corpus['loser_mask'].cpu().numpy()
    gap_other = other_gap[mask]

    def stats(x, name):
        x = np.asarray(x)
        if x.size == 0:
            print(f"  {name}: empty")
            return
        print(f"  {name} (n={x.size}): "
              f"min={x.min():.3f} p10={np.percentile(x,10):.3f} "
              f"p25={np.percentile(x,25):.3f} median={np.median(x):.3f} "
              f"p75={np.percentile(x,75):.3f} p90={np.percentile(x,90):.3f} "
              f"max={x.max():.3f}  mean={x.mean():.3f}")

    print("\n=== Logit-gap distributions ===")
    stats(gap_top1, "logit[winner] - logit[stored_top1]")
    stats(gap_other, "logit[winner] - logit[clean_other_loser]")

    # Margin candidates: what fraction is satisfied at various thresholds?
    print("\n=== Margin sensitivity (fraction of pairs ALREADY satisfied) ===")
    print("  threshold     top1 pairs    clean-other pairs")
    for m in [-1.0, -0.5, 0.0, 0.1, 0.25, 0.5, 1.0]:
        t1 = float((gap_top1 > m).mean())
        ot = float((gap_other > m).mean()) if gap_other.size else 0.0
        print(f"  {m:>+8.2f}       {t1:.3f}         {ot:.3f}")

    # Recommendation
    median_gap = float(np.median(gap_top1))
    print("\n=== Recommendation ===")
    if median_gap < -2.0:
        print(f"  Median top1 gap = {median_gap:.2f} (very negative).")
        print(f"  Default margin=0.25 means we ask the model to flip a "
              f"~2-logit gap. Heavy lift; consider lower λ or longer "
              f"warmup.")
    elif median_gap < -0.5:
        print(f"  Median top1 gap = {median_gap:.2f} (negative as expected). "
              f"Default margin=0.25 should be appropriate.")
    elif median_gap > 0.0:
        print(f"  Median top1 gap = {median_gap:.2f} (already positive!). "
              f"Either filter is too permissive or pillar3b already prefers "
              f"the winner. Re-check anchor filter.")
    print(f"  Suggested margin: 0.25 (default). Inspect distributions above "
          f"before locking in.")


if __name__ == '__main__':
    main()
