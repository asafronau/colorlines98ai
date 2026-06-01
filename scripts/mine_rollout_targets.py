"""Pillar 4 mining: direct-rollout soft policy targets (calibration probe).

Replaces MCTS visit distributions with rollout-derived soft targets on
stationary boundary states. Per ChatGPT 2026-05-24 design:

  - K=5 candidates (vs Phase 2's K=10; tighter focus, no noise tail)
  - R=48 common-RNG rollouts per (anchor, candidate)
  - H=100 horizon
  - Per-trajectory metrics:
      die_rate, leave_band_rate (P min_empty<30),
      lec_under_10_frac (fraction of trajectory turns with lec<10),
      score_rate (final score_gained / turns_played)
  - Weighted penalty:
      penalty = a*die + b*leave_band + c*lec_u10 + d*max(0, target_sr − sr)
  - Soft target: softmax(−penalty / τ) over the K candidates
  - Blended target: blend * rollout_target + (1−blend) * pillar3b_policy_top5

This is the calibration probe (default 200 anchors): diagnostics-first.
Phase 2 mining mistakes corrected:
  - Targets are SOFT (no margin/hinge); model can disagree at any margin
  - Targets include score_rate term so "stand still and don't die" is
    penalized; prevents the safety-only failure mode
  - Targets are blended with the original policy so the floor anchors don't
    catastrophically forget mean-game behavior

After mining: re-sweep (weights, τ, blend) on the raw rollout stats and
inspect resulting target shape WITHOUT re-rolling. Save raw rollouts to disk.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, set_start_method

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from game.board import ColorLinesGame
from game.rng import SimpleRng
from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation
from scripts.mine_stationary_counterfactuals import (
    policy_topk, restore_game, largest_empty_component,
    sample_anchors_from_selfplay,
)


# ── Per-worker globals ──────────────────────────────────────────────────────
_W_NET = None
_W_DEVICE = None
_W_DTYPE = None


def _init_worker(model_path, device_str):
    import torch as _t
    global _W_NET, _W_DEVICE, _W_DTYPE
    if device_str == 'cuda' and _t.cuda.is_available():
        _W_DEVICE = _t.device('cuda')
    elif device_str == 'mps' and _t.backends.mps.is_available():
        _W_DEVICE = _t.device('mps')
    else:
        _W_DEVICE = _t.device('cpu')
    _W_NET, _ = load_model(model_path, _W_DEVICE,
                            fp16=(_W_DEVICE.type != 'cpu'))
    _W_DTYPE = next(_W_NET.parameters()).dtype


def policy_argmax(net, device, net_dtype, game):
    out = policy_topk(net, device, net_dtype, game, k=1)
    return out[0][0] if out else None


def rollout_pillar4(net, device, net_dtype, anchor_state, first_move,
                    rollout_seed, H=100):
    """Common-RNG rollout. Tracks per-step:
      lec_under_10_frac — fraction of turns with lec<10 (sustained fragmentation)
      min_empty         — minimum empty count along trajectory
                          (used to compute leave_band_rate aggregate)
    """
    g = restore_game(anchor_state)
    g.rng = SimpleRng(rollout_seed)
    start_score = g.score
    res = g.move(*first_move)
    if not res['valid']:
        return None

    lec_under_10_count = 0
    turns_after_first = 1
    lec_now = largest_empty_component(g.board)
    if lec_now < 10:
        lec_under_10_count += 1
    min_empty = int((g.board == 0).sum())
    died = False
    for h in range(H - 1):
        if g.game_over:
            died = True
            break
        mv = policy_argmax(net, device, net_dtype, g)
        if mv is None:
            died = True
            break
        r = g.move(*mv)
        if not r['valid']:
            died = True
            break
        turns_after_first += 1
        lec_now = largest_empty_component(g.board)
        if lec_now < 10:
            lec_under_10_count += 1
        e = int((g.board == 0).sum())
        if e < min_empty:
            min_empty = e
    if g.game_over:
        died = True
    score_gained = int(g.score - start_score)
    return {
        'score_gained': score_gained,
        'died': died,
        'turns_played': turns_after_first,
        'lec_u10_frac': lec_under_10_count / max(turns_after_first, 1),
        'min_empty': min_empty,
        'score_rate': score_gained / max(turns_after_first, 1),
    }


def _worker_run(args):
    anchor_idx, rank, move, anchor_state, R, H = args
    out = []
    for rs in range(R):
        r = rollout_pillar4(_W_NET, _W_DEVICE, _W_DTYPE,
                             anchor_state, move, rs, H=H)
        out.append(r)
    return (anchor_idx, rank, out)


# ── Penalty / soft target ──────────────────────────────────────────────────


def compute_candidate_stats(rollouts):
    """Aggregate rollouts into per-candidate scalar stats."""
    rs = [r for r in rollouts if r is not None]
    if not rs:
        return None
    score_gained = np.array([r['score_gained'] for r in rs])
    turns = np.array([r['turns_played'] for r in rs])
    lec_u10 = np.array([r['lec_u10_frac'] for r in rs])
    min_empty = np.array([r['min_empty'] for r in rs])
    sr = np.array([r['score_rate'] for r in rs])
    died = np.array([r['died'] for r in rs])
    return {
        'n': len(rs),
        'die_rate': float(died.mean()),
        'leave_band_rate': float((min_empty < 30).mean()),
        'lec_u10_frac': float(lec_u10.mean()),
        'mean_score': float(score_gained.mean()),
        'p10_score': float(np.percentile(score_gained, 10)),
        'score_rate': float(sr.mean()),
        'p10_score_rate': float(np.percentile(sr, 10)),
        'mean_turns': float(turns.mean()),
    }


def penalty_per_candidate(stats, weights, target_sr=2.0):
    """Weighted penalty. Lower = better.

    weights: dict with keys {'die', 'leave_band', 'lec_u10', 'score_rate'}
    target_sr: score_rate target; deviation below is penalized.
    """
    score_term = max(0.0, target_sr - stats['score_rate'])
    return (weights['die'] * stats['die_rate']
            + weights['leave_band'] * stats['leave_band_rate']
            + weights['lec_u10'] * stats['lec_u10_frac']
            + weights['score_rate'] * score_term)


def build_soft_target(stats_per_cand, prior_per_cand, weights, tau, blend,
                      target_sr=2.0):
    """Return (rollout_target, blended_target) as numpy arrays length K."""
    K = len(stats_per_cand)
    penalties = np.array([penalty_per_candidate(s, weights, target_sr)
                           for s in stats_per_cand])
    logits = -penalties / max(tau, 1e-6)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    pol = np.array(prior_per_cand)
    pol = pol / pol.sum() if pol.sum() > 0 else np.ones(K) / K
    blended = blend * p + (1 - blend) * pol
    blended /= blended.sum()
    return p, blended


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',
                    default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--selfplay-dir', default='data/selfplay_v13')
    p.add_argument('--out',
                    default='alphatrain/data/rollout_targets_probe.pt')
    p.add_argument('--max-anchors', type=int, default=200)
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--R', type=int, default=48)
    p.add_argument('--H', type=int, default=100)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--device', default='cpu',
                    choices=['cpu', 'mps', 'cuda'])
    # Default penalty weights from ChatGPT 2026-05-24
    p.add_argument('--w-die', type=float, default=5.0)
    p.add_argument('--w-leave-band', type=float, default=1.0)
    p.add_argument('--w-lec-u10', type=float, default=0.5)
    p.add_argument('--w-score-rate', type=float, default=0.2)
    p.add_argument('--target-sr', type=float, default=2.0)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--blend', type=float, default=0.7,
                    help='blend * rollout + (1-blend) * pillar3b_policy_top5')
    args = p.parse_args()

    weights = {
        'die': args.w_die,
        'leave_band': args.w_leave_band,
        'lec_u10': args.w_lec_u10,
        'score_rate': args.w_score_rate,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Pillar 4 mining (probe: {args.max_anchors} anchors) ===",
          flush=True)
    print(f"  K={args.top_k}  R={args.R}  H={args.H}", flush=True)
    print(f"  weights={weights}  target_sr={args.target_sr}", flush=True)
    print(f"  tau={args.tau}  blend={args.blend}", flush=True)

    anchors = sample_anchors_from_selfplay(args.selfplay_dir,
                                             args.max_anchors)
    n_anchors = len(anchors)
    print(f"  Sampled {n_anchors} anchors", flush=True)
    if n_anchors == 0:
        print("No anchors — aborting")
        return

    # Top-K candidates per anchor (main process, fast)
    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    net, _ = load_model(args.model, device,
                         fp16=(device.type != 'cpu'))
    net_dtype = next(net.parameters()).dtype

    anchor_topks = []
    t0 = time.time()
    for i, anchor in enumerate(anchors):
        g = restore_game(anchor)
        topk = policy_topk(net, device, net_dtype, g, k=args.top_k)
        anchor_topks.append(topk)
        if (i + 1) % 50 == 0:
            print(f"  topk [{i+1}/{n_anchors}] {time.time() - t0:.0f}s",
                  flush=True)
    print(f"  Top-K precompute: {time.time() - t0:.0f}s", flush=True)

    del net
    if device.type == 'mps':
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    # Build work units
    units = []
    for ai, (anchor, topk) in enumerate(zip(anchors, anchor_topks)):
        for rank, (move, _prob) in enumerate(topk):
            units.append((ai, rank, move, anchor, args.R, args.H))
    print(f"\n=== Rollouts: {len(units)} units × R={args.R} = "
          f"{len(units) * args.R} rollouts ===", flush=True)

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    raw_rollouts = defaultdict(dict)
    t0 = time.time()
    with Pool(processes=args.workers,
              initializer=_init_worker,
              initargs=(args.model, args.device)) as pool:
        n_done = 0
        for anchor_idx, rank, rollouts in pool.imap_unordered(
                _worker_run, units, chunksize=2):
            raw_rollouts[anchor_idx][rank] = rollouts
            n_done += 1
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / n_done * (len(units) - n_done)
                print(f"  rollouts [{n_done}/{len(units)}] "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    # Aggregate per-anchor
    print(f"\n=== Aggregating ===", flush=True)
    records = []
    for ai, (anchor, topk) in enumerate(zip(anchors, anchor_topks)):
        stats_per_cand = []
        prior_per_cand = []
        moves = []
        for rank, (move, prior) in enumerate(topk):
            rollouts = raw_rollouts[ai].get(rank, [])
            stats = compute_candidate_stats(rollouts)
            if stats is None:
                continue
            stats_per_cand.append(stats)
            prior_per_cand.append(prior)
            moves.append(move)
        if len(stats_per_cand) < 2:
            continue
        rollout_target, blended_target = build_soft_target(
            stats_per_cand, prior_per_cand, weights, args.tau, args.blend,
            target_sr=args.target_sr)
        records.append({
            'anchor': anchor,
            'moves': moves,
            'priors': prior_per_cand,
            'stats': stats_per_cand,
            'rollout_target': rollout_target.tolist(),
            'blended_target': blended_target.tolist(),
        })

    print(f"  {len(records)} valid records (of {n_anchors})", flush=True)

    torch.save({
        'records': records,
        'config': {
            'max_anchors': args.max_anchors,
            'top_k': args.top_k,
            'R': args.R,
            'H': args.H,
            'model': args.model,
            'selfplay_dir': args.selfplay_dir,
            'weights': weights,
            'target_sr': args.target_sr,
            'tau': args.tau,
            'blend': args.blend,
        },
    }, args.out)
    sz = os.path.getsize(args.out) / 1e6
    print(f"  saved {args.out} ({sz:.1f} MB)", flush=True)

    # ── Diagnostics ────────────────────────────────────────────────────
    print(f"\n=== Diagnostics ===", flush=True)
    rollout_targets = np.array([r['rollout_target'] for r in records])
    blended_targets = np.array([r['blended_target'] for r in records])

    def entropy(p_batch):
        eps = 1e-9
        return -np.sum(p_batch * np.log(p_batch + eps), axis=1)

    e_rollout = entropy(rollout_targets)
    e_blended = entropy(blended_targets)
    e_uniform = np.log(args.top_k)
    print(f"  Entropy (uniform K={args.top_k}: {e_uniform:.3f})")
    print(f"    rollout target: mean={e_rollout.mean():.3f} "
          f"p10={np.percentile(e_rollout,10):.3f} "
          f"p90={np.percentile(e_rollout,90):.3f}")
    print(f"    blended target: mean={e_blended.mean():.3f} "
          f"p10={np.percentile(e_blended,10):.3f} "
          f"p90={np.percentile(e_blended,90):.3f}")

    # Top-1 disagreement vs policy
    rollout_top1 = rollout_targets.argmax(axis=1)
    blended_top1 = blended_targets.argmax(axis=1)
    # policy_top1 is rank 0 since topk returns sorted by prior
    pol_top1_rate = float((rollout_top1 == 0).mean())
    blended_pol_top1_rate = float((blended_top1 == 0).mean())
    print(f"  Top1 = policy_top1:")
    print(f"    rollout-only:  {pol_top1_rate:.3f} "
          f"(disagree {1-pol_top1_rate:.3f})")
    print(f"    blended:       {blended_pol_top1_rate:.3f} "
          f"(disagree {1-blended_pol_top1_rate:.3f})")
    print(f"  ChatGPT target: disagree in 0.40-0.70 range")

    # Per-term distributions
    flat = [s for r in records for s in r['stats']]
    all_die = np.array([s['die_rate'] for s in flat])
    all_lb = np.array([s['leave_band_rate'] for s in flat])
    all_lec = np.array([s['lec_u10_frac'] for s in flat])
    all_sr = np.array([s['score_rate'] for s in flat])
    print(f"  Per-candidate metrics (n={len(flat)}):")

    def stats(x, name):
        print(f"    {name}: min={x.min():.3f} p25={np.percentile(x,25):.3f} "
              f"med={np.median(x):.3f} p75={np.percentile(x,75):.3f} "
              f"max={x.max():.3f} mean={x.mean():.3f}")

    stats(all_die, 'die_rate        ')
    stats(all_lb,  'leave_band_rate ')
    stats(all_lec, 'lec_u10_frac    ')
    stats(all_sr,  'score_rate      ')

    # Penalty term contributions to total penalty
    contrib_die = weights['die'] * all_die
    contrib_lb = weights['leave_band'] * all_lb
    contrib_lec = weights['lec_u10'] * all_lec
    contrib_sr = weights['score_rate'] * np.maximum(0, args.target_sr - all_sr)
    print(f"  Penalty contributions (weighted, median):")
    print(f"    die={np.median(contrib_die):.3f} "
          f"leave_band={np.median(contrib_lb):.3f} "
          f"lec_u10={np.median(contrib_lec):.3f} "
          f"score_rate={np.median(contrib_sr):.3f}")


if __name__ == '__main__':
    main()
