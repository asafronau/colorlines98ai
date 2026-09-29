"""Sample real game states from `eval --record-dir` recordings into an anchors.h file, for CPU benchmarks
and equivalence tests of the C++ game loop (build/eval_cpu_bench).

    python -m alphatrain.scripts.export_bench_states --games-dir alphatrain/data/greedy_A2_cap100k \
        --out alphatrain/inference_cpp/data/bench_states.txt --n 4000
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    files = sorted(glob.glob(os.path.join(a.games_dir, 'game_seed*.json')))
    picks = rng.choice(len(files), size=a.n, replace=len(files) < a.n)
    with open(a.out, 'w') as out:
        for i, fi in enumerate(picks):
            states = json.load(open(files[fi]))['states']
            s = states[rng.integers(len(states))]
            cells = ' '.join(str(v) for row in s['board'] for v in row)
            nb = [f"{b['row']} {b['col']} {b['color']}" for b in s['next_balls']][:3]
            nb += ['-1 -1 -1'] * (3 - len(nb))
            out.write(f"{1000 + i} {s['turn']} 1000000 {cells} {' '.join(nb)}\n")
    print(f'wrote {a.n} states from {len(files)} games to {a.out}')


if __name__ == '__main__':
    main()
