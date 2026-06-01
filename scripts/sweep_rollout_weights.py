"""Sweep (weights, tau, blend) on the cached rollout probe data.

Reads alphatrain/data/rollout_targets_probe.pt (raw per-candidate stats), then
recomputes soft targets for a grid of configurations and reports diagnostics.

No re-mining needed — this is purely post-hoc.
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mine_rollout_targets import (
    build_soft_target, penalty_per_candidate)


def diagnose(records, weights, tau, blend, target_sr=2.0):
    rollout_targets = []
    blended_targets = []
    for r in records:
        rt, bt = build_soft_target(
            r['stats'], r['priors'], weights, tau, blend, target_sr=target_sr)
        rollout_targets.append(rt)
        blended_targets.append(bt)
    rt = np.array(rollout_targets)
    bt = np.array(blended_targets)

    def entropy(p):
        eps = 1e-9
        return -np.sum(p * np.log(p + eps), axis=1)

    K = rt.shape[1]
    er = entropy(rt)
    eb = entropy(bt)
    # Top1 disagreement with policy (rank 0 by prior in topk ordering)
    rt_top1 = rt.argmax(axis=1)
    bt_top1 = bt.argmax(axis=1)
    rt_disagree = float((rt_top1 != 0).mean())
    bt_disagree = float((bt_top1 != 0).mean())
    # How peaked is the top1 (mean & p10)
    rt_top1_p = rt.max(axis=1)
    bt_top1_p = bt.max(axis=1)
    return {
        'rt_entropy_mean': er.mean(), 'rt_entropy_p10': np.percentile(er, 10),
        'bt_entropy_mean': eb.mean(), 'bt_entropy_p10': np.percentile(eb, 10),
        'rt_disagree': rt_disagree,
        'bt_disagree': bt_disagree,
        'rt_top1_p_mean': rt_top1_p.mean(),
        'bt_top1_p_mean': bt_top1_p.mean(),
        'rt_top1_p_p10': np.percentile(rt_top1_p, 10),
        'bt_top1_p_p10': np.percentile(bt_top1_p, 10),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe',
                    default='alphatrain/data/rollout_targets_probe.pt')
    p.add_argument('--target-sr', type=float, default=2.0)
    args = p.parse_args()

    data = torch.load(args.probe, map_location='cpu', weights_only=False)
    records = data['records']
    print(f"Loaded {len(records)} records from {args.probe}", flush=True)
    print(f"  Uniform entropy K=5: {np.log(5):.3f}", flush=True)
    print(f"  ChatGPT goldilocks: top1 disagree 40-70%, entropy 1.0-1.4", flush=True)
    print()

    # Sweep grid. Focus on leave_band since the probe showed it's the only
    # signal carrier; die/lec_u10/score_rate are near-zero contributors.
    sweep = []
    for w_lb in [1.0, 3.0, 5.0, 10.0]:
        for tau in [0.2, 0.3, 0.5, 1.0]:
            for blend in [0.5, 0.7, 0.9]:
                sweep.append((w_lb, tau, blend))

    print(f"{'w_lb':>5} {'tau':>5} {'blend':>6}  "
          f"{'rt_ent':>7} {'bt_ent':>7}  "
          f"{'rt_disag':>9} {'bt_disag':>9}  "
          f"{'rt_top1':>8} {'bt_top1':>8}")
    print('-' * 92)
    for w_lb, tau, blend in sweep:
        weights = {'die': 5.0, 'leave_band': w_lb,
                   'lec_u10': 0.5, 'score_rate': 0.2}
        d = diagnose(records, weights, tau, blend, args.target_sr)
        ok = ""
        if 0.40 <= d['bt_disagree'] <= 0.70 and 1.0 <= d['bt_entropy_mean'] <= 1.4:
            ok = "  ◀ goldilocks"
        print(f"{w_lb:>5.1f} {tau:>5.2f} {blend:>6.2f}  "
              f"{d['rt_entropy_mean']:>7.3f} {d['bt_entropy_mean']:>7.3f}  "
              f"{d['rt_disagree']:>9.3f} {d['bt_disagree']:>9.3f}  "
              f"{d['rt_top1_p_mean']:>8.3f} {d['bt_top1_p_mean']:>8.3f}"
              f"{ok}", flush=True)


if __name__ == '__main__':
    main()
