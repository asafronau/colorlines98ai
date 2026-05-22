"""Audit what settings each crisis JSON was actually generated with.

We store `replay_sims` per-file but NOT the rewind (recovery-turns /
prevention-turns). So this script:
  1. Shows the actual distribution of `replay_sims` (recovery + prevention).
  2. Bins files by mtime to detect setting changes across batches.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from datetime import datetime

import numpy as np


def _scan(directory, label):
    files = sorted(glob.glob(
        os.path.join(directory, f'game_seed*_{label}_*.json')))
    rows = []
    for f in files:
        try:
            with open(f) as fp:
                g = json.load(fp)
            rows.append({
                'path': f,
                'mtime': os.path.getmtime(f),
                'replay_sims': g.get('replay_sims'),
                'replay_from_turn': g.get('replay_from_turn'),
                'turns': g.get('turns'),
                'capped': g.get('capped'),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return rows


def _report(name, rows):
    print(f"\n=== {name}: n={len(rows)} ===")
    if not rows:
        return
    sims = Counter(r['replay_sims'] for r in rows)
    print("  replay_sims distribution:")
    for s, c in sorted(sims.items(), key=lambda x: -x[1]):
        print(f"    sims={s}: {c} files ({100*c/len(rows):.1f}%)")

    # Sort by mtime, bin into 6 chronological buckets, show settings per bucket
    rows = sorted(rows, key=lambda r: r['mtime'])
    n_buckets = 6
    bsize = max(1, len(rows) // n_buckets)
    print(f"  chronological buckets (by mtime):")
    for i in range(0, len(rows), bsize):
        chunk = rows[i:i + bsize]
        if not chunk:
            continue
        sims_in = Counter(r['replay_sims'] for r in chunk)
        sims_str = ', '.join(f"sims={s}: {c}"
                              for s, c in sorted(sims_in.items()))
        t_lo = datetime.fromtimestamp(chunk[0]['mtime']).strftime("%m-%d %H:%M")
        t_hi = datetime.fromtimestamp(chunk[-1]['mtime']).strftime("%m-%d %H:%M")
        cap_rate = np.mean([r['capped'] for r in chunk])
        r_from = np.median([r['replay_from_turn'] for r in chunk])
        print(f"    {t_lo} → {t_hi}: {sims_str:30s} cap={100*cap_rate:.0f}%  "
              f"median replay_from={r_from:.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--crisis-dir', default='data/crisis_v13')
    args = p.parse_args()

    for label in ['recovery', 'prevention']:
        rows = _scan(args.crisis_dir, label)
        _report(f'{label.upper()} ({args.crisis_dir})', rows)

    print("\nNOTE: recovery-turns / prevention-turns (rewind values) are NOT")
    print("stored per-file. Inferable only from process logs or by pairing")
    print("recovery+prevention from the same original_seed (gives rewind diff,")
    print("not absolute). Recommend adding to JSON for future runs.")


if __name__ == '__main__':
    main()
