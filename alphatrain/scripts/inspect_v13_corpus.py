"""Quick inspection of in-progress V13 corpus.

Reports score / turn / state-count distributions for selfplay_v13 and
crisis_v13 game JSON dirs. Helps verify data quality during the long
mining wall-clock without waiting for full tensor build.

Usage:
    python -m alphatrain.scripts.inspect_v13_corpus \\
        --selfplay-dir data/selfplay_v13 \\
        --crisis-dir data/crisis_v13
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _scores_from_filenames(d):
    """Filenames encode scores: game_seed{N}_score{S}.json or
    game_seed{N}_{recovery|prevention}_score{S}.json."""
    files = sorted(glob.glob(os.path.join(d, 'game_seed*.json')))
    scores = []
    for f in files:
        try:
            s = int(f.split('score')[1].split('.json')[0])
            scores.append(s)
        except (IndexError, ValueError):
            continue
    return files, scores


def _state_count(path):
    try:
        with open(path) as fp:
            g = json.load(fp)
        return (len(g.get('moves', [])), g.get('turns', None),
                g.get('capped', None), g.get('replay_from_turn', None))
    except (json.JSONDecodeError, OSError):
        return None, None, None, None


def _report(name, files, scores, sample_n=50):
    print(f"\n=== {name} ===")
    print(f"  files: {len(files)}")
    if not files:
        return
    s = np.asarray(scores)
    print(f"  score: mean={s.mean():.0f}  median={np.median(s):.0f}")
    print(f"  P5={np.percentile(s, 5):.0f}  P10={np.percentile(s, 10):.0f}  "
          f"P25={np.percentile(s, 25):.0f}  "
          f"P75={np.percentile(s, 75):.0f}  P90={np.percentile(s, 90):.0f}  "
          f"P95={np.percentile(s, 95):.0f}  max={s.max()}")
    print(f"  <1000: {(s < 1000).sum()} ({100*(s < 1000).sum()/len(s):.1f}%)  "
          f">10K: {(s > 10000).sum()} ({100*(s > 10000).sum()/len(s):.1f}%)  "
          f">20K: {(s > 20000).sum()} ({100*(s > 20000).sum()/len(s):.1f}%)  "
          f">30K: {(s > 30000).sum()} ({100*(s > 30000).sum()/len(s):.1f}%)")

    # Sample state counts (the actually-useful diagnostic for crisis)
    rng = np.random.default_rng(0)
    sample_idx = (rng.choice(len(files), size=min(sample_n, len(files)),
                              replace=False)
                  if len(files) > sample_n else np.arange(len(files)))
    n_states = []
    n_turns = []
    n_survived = []  # continue_turns (turns - replay_from_turn); crisis only
    n_capped = 0
    n_zero = 0
    for i in sample_idx:
        ns, nt, cap, rfrom = _state_count(files[i])
        if ns is None:
            continue
        n_states.append(ns)
        if nt is not None:
            n_turns.append(nt)
            if rfrom is not None:
                n_survived.append(nt - rfrom)
        if cap:
            n_capped += 1
        if ns == 0:
            n_zero += 1
    if n_states:
        arr = np.asarray(n_states)
        print(f"  states/file (sample of {len(n_states)}): "
              f"mean={arr.mean():.0f}  median={np.median(arr):.0f}  "
              f"min={arr.min()}  max={arr.max()}  "
              f"zero-state files: {n_zero}/{len(n_states)} "
              f"({100*n_zero/len(n_states):.1f}%)")
        total_est = int(arr.mean() * len(files))
        print(f"  estimated TOTAL states in {name}: ~{total_est:,}")
    if n_turns:
        tarr = np.asarray(n_turns)
        print(f"  turns/file: mean={tarr.mean():.0f}  median={np.median(tarr):.0f}")
    if n_survived:
        sv = np.asarray(n_survived)
        print(f"  continue_turns survived (post-rewind): "
              f"mean={sv.mean():.0f}  median={np.median(sv):.0f}  "
              f"P5={np.percentile(sv, 5):.0f}  P10={np.percentile(sv, 10):.0f}  "
              f"P25={np.percentile(sv, 25):.0f}  "
              f"P75={np.percentile(sv, 75):.0f}  max={sv.max()}")
    print(f"  capped: {n_capped}/{len(sample_idx)} ({100*n_capped/max(1, len(sample_idx)):.0f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selfplay-dir', default='data/selfplay_v13')
    p.add_argument('--crisis-dir', default='data/crisis_v13')
    p.add_argument('--sample-n', type=int, default=80,
                   help='Random JSONs to fully inspect for state counts.')
    args = p.parse_args()

    sp_files, sp_scores = _scores_from_filenames(args.selfplay_dir)
    _report(f'SELFPLAY ({args.selfplay_dir})', sp_files, sp_scores, args.sample_n)

    if not os.path.isdir(args.crisis_dir):
        print(f"\n{args.crisis_dir}: not a directory; skipping")
        return

    # Split crisis files by recovery vs prevention
    cr_files = sorted(glob.glob(os.path.join(args.crisis_dir,
                                              'game_seed*.json')))
    rec_files = [f for f in cr_files if 'recovery' in f]
    prev_files = [f for f in cr_files if 'prevention' in f]
    rec_scores = [int(f.split('score')[1].split('.json')[0])
                   for f in rec_files]
    prev_scores = [int(f.split('score')[1].split('.json')[0])
                    for f in prev_files]
    _report(f'CRISIS recovery ({args.crisis_dir})',
             rec_files, rec_scores, args.sample_n)
    _report(f'CRISIS prevention ({args.crisis_dir})',
             prev_files, prev_scores, args.sample_n)


if __name__ == '__main__':
    main()
