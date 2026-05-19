"""Overnight trajectory analysis for Path B Runs A and C.

For each checkpoint, computes:
  - Distribution stats: entropy (full softmax), top1 confidence, oracle_best
    global rank in 6561 logits
  - Oracle metrics: KL_w, top1 by Delta-cap bucket, P(oracle_best) on top-6,
    logit_gap by bucket
  - Gameplay (policy-only): mean, P10/P50/P75/P90/P95, <500/<1000/>5000/>10000
    rates (500 seeds, max_turns 8000)
  - From checkpoint metadata: V12 val_loss

Output (saved incrementally so interrupts are safe):
  - <output>.json
  - <output>.md (final table)

Robust: per-checkpoint try/except so one failure doesn't kill the run.
Resumable: re-running with same --output reads existing JSON, skips done ckpts.

Usage:
    nohup python -m alphatrain.scripts.overnight_analysis \\
        --checkpoints a_smoke_epoch_5 a_smoke_epoch_7 a_smoke_epoch_11 \\
                       a_smoke_epoch_12 c_smoke_epoch_5 c_smoke_epoch_7 \\
                       c_smoke_epoch_11 c_smoke_epoch_12 \\
        --output alphatrain/data/overnight_results.json \\
        --seeds 500 > overnight.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from alphatrain.scripts.analyze_path_b_checkpoint import (
    gather_metrics, load_model)
from alphatrain.train_path_b import OracleDataset, reliability_weight


# Match the reliability ramp buckets in analyze_path_b_checkpoint.
BUCKETS = [
    ('noise',    0.00, 0.05),
    ('weak',     0.05, 0.10),
    ('medium',   0.10, 0.15),
    ('strong',   0.15, 0.25),
    ('dominant', 0.25, 1.001),
]


def parse_gameplay_stdout(text):
    """Parse the alphatrain.scripts.eval_parallel stdout."""
    out = {}
    m = re.search(r'MEAN\s*\|\s*(\d+)', text)
    if m:
        out['mean'] = int(m.group(1))
    m = re.search(
        r'P1=(\d+)\s+P5=(\d+)\s+P10=(\d+)\s+P25=(\d+)\s+P50=(\d+)\s+'
        r'P75=(\d+)\s+P90=(\d+)\s+P95=(\d+)', text)
    if m:
        for i, key in enumerate(['p1', 'p5', 'p10', 'p25', 'p50',
                                   'p75', 'p90', 'p95']):
            out[key] = int(m.group(i + 1))
    m = re.search(
        r'<500:\s*(\d+).*?<1000:\s*(\d+).*?>5000:\s*(\d+).*?>10000:\s*(\d+)',
        text, re.DOTALL)
    if m:
        out['lt_500'] = int(m.group(1))
        out['lt_1000'] = int(m.group(2))
        out['gt_5000'] = int(m.group(3))
        out['gt_10000'] = int(m.group(4))
    return out


def run_gameplay(ckpt_path, device, n_seeds, workers, max_turns,
                  timeout=3600):
    """Spawn eval_parallel subprocess; return parsed metrics or None."""
    cmd = [
        sys.executable, '-m', 'alphatrain.scripts.eval_parallel',
        '--model', ckpt_path,
        '--policy-only',
        '--device', device,
        '--workers', str(workers),
        '--max-turns', str(max_turns),
        '--seeds', *[str(s) for s in range(n_seeds)],
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    parsed = parse_gameplay_stdout(proc.stdout + proc.stderr)
    parsed['elapsed_s'] = elapsed
    parsed['returncode'] = proc.returncode
    parsed['stderr_tail'] = proc.stderr[-500:] if proc.stderr else ''
    return parsed


def run_distribution_analysis(ckpt_path, oracle_path, device, batch_size=4096,
                                num_blocks=10, channels=256):
    """In-process distribution + oracle metrics; return aggregates."""
    model, epoch, val_loss, _ = load_model(
        ckpt_path, num_blocks, channels, device)

    ods = OracleDataset(oracle_path, device=device)
    _, val_idx = ods.split(val_frac=0.05, seed=2026)

    (dcap, gap, p, kl, agree, ent_full, p_top1_full,
     global_rank) = gather_metrics(
        model, ods, val_idx, device,
        beta=10.0, noise_floor=0.05, scale=0.20,
        batch_size=batch_size)

    w = reliability_weight(dcap, 0.05, 0.20)

    out = {
        'epoch': int(epoch) + 1 if isinstance(epoch, int) else None,
        'val_loss': float(val_loss),
        'n_val': int(dcap.numel()),
        # Full distribution
        'entropy_full_mean': float(ent_full.mean().item()),
        'entropy_full_median': float(ent_full.median().item()),
        'p_top1_full_mean': float(p_top1_full.mean().item()),
        'p_top1_full_median': float(p_top1_full.median().item()),
        # Oracle aggregates
        'kl_mean': float(kl.mean().item()),
        'kl_weighted': float(((w * kl).sum() / w.sum().clamp(min=1.0)).item()),
        'top1_agree_all': float(agree.float().mean().item()),
        'p_oracle_top6_mean': float(p.mean().item()),
        'logit_gap_mean': float(gap.mean().item()),
        # Global rank (over all 6561 actions)
        'global_rank_mean': float(global_rank.float().mean().item()),
        'global_rank_median': float(global_rank.float().median().item()),
        'global_rank_p90': float(np.percentile(global_rank.numpy(), 90)),
    }

    # Per-bucket: top1, logit_gap, p_oracle, global_rank, n
    per_bucket = {}
    for name, lo, hi in BUCKETS:
        m = (dcap >= lo) & (dcap < hi)
        n = int(m.sum().item())
        if n == 0:
            per_bucket[name] = {'n': 0}
            continue
        per_bucket[name] = {
            'n': n,
            'top1': float(agree[m].float().mean().item()),
            'logit_gap': float(gap[m].mean().item()),
            'p_oracle_top6': float(p[m].mean().item()),
            'global_rank_mean': float(global_rank[m].float().mean().item()),
            'global_rank_median': float(global_rank[m].float().median().item()),
        }
    out['by_bucket'] = per_bucket
    return out


def save_json(results, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp, path)


def emit_markdown(results, path):
    rows = []
    for ckpt, r in results.items():
        if 'error' in r and 'distribution' not in r:
            continue
        d = r.get('distribution', {})
        g = r.get('gameplay', {})
        b15 = d.get('by_bucket', {}).get('strong', {})
        b25 = d.get('by_bucket', {}).get('dominant', {})

        lt1000_pct = None
        if g.get('lt_1000') is not None:
            n_seeds = 500
            lt1000_pct = 100.0 * g['lt_1000'] / n_seeds

        rows.append({
            'ckpt': ckpt,
            'epoch': d.get('epoch'),
            'val_loss': d.get('val_loss'),
            'mean': g.get('mean'),
            'p10': g.get('p10'),
            'p50': g.get('p50'),
            'p75': g.get('p75'),
            'p95': g.get('p95'),
            'lt1000_pct': lt1000_pct,
            'entropy': d.get('entropy_full_mean'),
            'top1_conf': d.get('p_top1_full_mean'),
            'kl_w': d.get('kl_weighted'),
            'top1_all': d.get('top1_agree_all'),
            'global_rank_med': d.get('global_rank_median'),
            'top1_d15': b15.get('top1') if b15.get('n') else None,
            'top1_d25': b25.get('top1') if b25.get('n') else None,
        })

    lines = []
    lines.append('# Path B trajectory analysis')
    lines.append('')
    lines.append('| ckpt | epoch | val_loss | mean | P10 | P50 | P75 | '
                 'P95 | <1000% | entropy | top1_conf | KL_w | top1_all | '
                 'global_rank_med | top1>=.15 | top1>=.25 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        cells = [
            r['ckpt'],
            r['epoch'] or '-',
            f"{r['val_loss']:.4f}" if r['val_loss'] is not None else '-',
            r['mean'] if r['mean'] is not None else '-',
            r['p10'] if r['p10'] is not None else '-',
            r['p50'] if r['p50'] is not None else '-',
            r['p75'] if r['p75'] is not None else '-',
            r['p95'] if r['p95'] is not None else '-',
            f"{r['lt1000_pct']:.1f}" if r['lt1000_pct'] is not None else '-',
            f"{r['entropy']:.3f}" if r['entropy'] is not None else '-',
            f"{r['top1_conf']:.3f}" if r['top1_conf'] is not None else '-',
            f"{r['kl_w']:.3f}" if r['kl_w'] is not None else '-',
            f"{r['top1_all']:.3f}" if r['top1_all'] is not None else '-',
            f"{r['global_rank_med']:.1f}" if r['global_rank_med'] is not None else '-',
            f"{r['top1_d15']:.3f}" if r['top1_d15'] is not None else '-',
            f"{r['top1_d25']:.3f}" if r['top1_d25'] is not None else '-',
        ]
        lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help='Checkpoint stems (without .pt) under data-dir.')
    p.add_argument('--data-dir', default='alphatrain/data')
    p.add_argument('--oracle-tensor',
                   default='alphatrain/data/phase1_oracle_path_b.pt')
    p.add_argument('--output', required=True,
                   help='JSON file for results. Sibling .md also written.')
    p.add_argument('--device', default='mps')
    p.add_argument('--seeds', type=int, default=500,
                   help='Number of gameplay seeds (default 500).')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--max-turns', type=int, default=8000)
    p.add_argument('--skip-gameplay', action='store_true')
    p.add_argument('--skip-distribution', action='store_true')
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoints: {len(args.checkpoints)}", flush=True)
    print(f"Seeds per gameplay run: {args.seeds}", flush=True)
    print(f"Output: {args.output}", flush=True)

    # Resumability
    results = {}
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                results = json.load(f)
            done = [k for k, v in results.items()
                     if 'distribution' in v and 'gameplay' in v
                     and 'error' not in v]
            print(f"  Resuming: {len(done)} checkpoints already done.",
                  flush=True)
        except Exception:
            print(f"  Existing JSON unreadable; starting fresh.", flush=True)

    t_start = time.time()

    for i, ckpt_stem in enumerate(args.checkpoints):
        ckpt_path = os.path.join(args.data_dir, ckpt_stem + '.pt')
        print(f"\n{'='*72}\n[{i+1}/{len(args.checkpoints)}] {ckpt_stem}\n"
              f"{'='*72}", flush=True)
        if not os.path.exists(ckpt_path):
            print(f"  MISSING: {ckpt_path}", flush=True)
            results[ckpt_stem] = {'error': f'missing checkpoint {ckpt_path}'}
            save_json(results, args.output)
            continue

        prior = results.get(ckpt_stem, {})
        has_dist = 'distribution' in prior and 'error' not in prior
        has_game = 'gameplay' in prior and 'error' not in prior
        if has_dist and has_game:
            print(f"  Already complete; skipping.", flush=True)
            continue

        results.setdefault(ckpt_stem, {})
        results[ckpt_stem]['started_at'] = datetime.datetime.now().isoformat()

        # Distribution analysis (in-process, fast)
        if not args.skip_distribution and not has_dist:
            try:
                t0 = time.time()
                dist = run_distribution_analysis(
                    ckpt_path, args.oracle_tensor, device)
                dist['elapsed_s'] = time.time() - t0
                results[ckpt_stem]['distribution'] = dist
                print(f"  Distribution: {time.time()-t0:.0f}s; "
                      f"val_loss={dist['val_loss']:.4f}, "
                      f"entropy={dist['entropy_full_mean']:.3f}, "
                      f"top1_conf={dist['p_top1_full_mean']:.3f}, "
                      f"KL_w={dist['kl_weighted']:.4f}, "
                      f"global_rank_med={dist['global_rank_median']:.1f}",
                      flush=True)
                save_json(results, args.output)
            except Exception as e:
                msg = f"distribution failed: {e}\n{traceback.format_exc()}"
                print(f"  ERROR: {msg}", flush=True)
                results[ckpt_stem]['error'] = msg
                save_json(results, args.output)

        # Gameplay (subprocess, slow)
        if not args.skip_gameplay and not has_game:
            try:
                t0 = time.time()
                game = run_gameplay(
                    ckpt_path, args.device, args.seeds, args.workers,
                    args.max_turns)
                results[ckpt_stem]['gameplay'] = game
                print(f"  Gameplay: {time.time()-t0:.0f}s; "
                      f"mean={game.get('mean', '?')}, "
                      f"P50={game.get('p50', '?')}, "
                      f"P95={game.get('p95', '?')}", flush=True)
                save_json(results, args.output)
            except Exception as e:
                msg = f"gameplay failed: {e}\n{traceback.format_exc()}"
                print(f"  ERROR: {msg}", flush=True)
                results[ckpt_stem]['error'] = msg
                save_json(results, args.output)

        results[ckpt_stem]['completed_at'] = datetime.datetime.now().isoformat()
        save_json(results, args.output)

    md_path = (args.output[:-5] + '.md'
                if args.output.endswith('.json') else args.output + '.md')
    emit_markdown(results, md_path)
    print(f"\n{'='*72}\nDONE in {(time.time()-t_start)/60:.1f}min.\n"
          f"JSON: {args.output}\nMarkdown: {md_path}", flush=True)


if __name__ == '__main__':
    main()
