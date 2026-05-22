"""Infer prevention rewind delta from paired recovery/prevention files.

If recovery_rewind is known (= 15 for both V12 and V13), then for any
original_seed that has BOTH a recovery and prevention file:
  prevention_rewind = recovery_rewind +
                      (recovery_replay_from - prevention_replay_from)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict


def _pair(directory, recovery_rewind=15):
    files = sorted(glob.glob(os.path.join(directory, 'game_seed*.json')))
    by_orig = defaultdict(dict)
    for f in files:
        try:
            with open(f) as fp:
                g = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        os_ = g.get('original_seed')
        label = g.get('label')
        rfrom = g.get('replay_from_turn')
        if os_ is None or label not in ('recovery', 'prevention'):
            continue
        by_orig[os_][label] = rfrom
        by_orig[os_][f'mtime_{label[0]}'] = os.path.getmtime(f)

    # Carry mtime alongside for chronological bucketing
    deltas = []
    deltas_with_mtime = []
    for os_, d in by_orig.items():
        if 'recovery' in d and 'prevention' in d:
            deltas.append(d['recovery'] - d['prevention'])
            t = max(d['mtime_r'], d['mtime_p']) if 'mtime_r' in d else 0
            deltas_with_mtime.append((t, d['recovery'] - d['prevention']))

    print(f"  paired files: {len(deltas)}")
    if not deltas:
        return
    # delta = recovery_replay_from - prevention_replay_from
    # delta should equal prevention_rewind - recovery_rewind
    # so prevention_rewind = delta + recovery_rewind
    c = Counter(deltas)
    print(f"  top (recovery_replay_from - prevention_replay_from) values:")
    for d, n in c.most_common(5):
        prevention_rewind = d + recovery_rewind
        pct = 100 * n / len(deltas)
        print(f"    delta={d:4d}  →  prevention_rewind={prevention_rewind}  "
              f"({n} pairs, {pct:.1f}%)")

    # Chronological buckets: did the rewind setting change across batches?
    deltas_with_mtime.sort()
    n_buckets = 6
    bsize = max(1, len(deltas_with_mtime) // n_buckets)
    print(f"  per-batch (mtime-sorted, 6 chronological buckets):")
    from datetime import datetime
    for i in range(0, len(deltas_with_mtime), bsize):
        chunk = deltas_with_mtime[i:i + bsize]
        if not chunk:
            continue
        cc = Counter(d for _, d in chunk)
        t_lo = datetime.fromtimestamp(chunk[0][0]).strftime("%m-%d %H:%M")
        t_hi = datetime.fromtimestamp(chunk[-1][0]).strftime("%m-%d %H:%M")
        deltas_str = ', '.join(f"d={k} ({v})" for k, v in cc.most_common(3))
        print(f"    {t_lo} → {t_hi}: {deltas_str}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v12-dir', default='data/crisis_v12')
    p.add_argument('--v13-dir', default='data/crisis_v13')
    args = p.parse_args()

    print(f"=== V12 ({args.v12_dir}) ===")
    _pair(args.v12_dir)
    print(f"\n=== V13 ({args.v13_dir}) ===")
    _pair(args.v13_dir)


if __name__ == '__main__':
    main()
