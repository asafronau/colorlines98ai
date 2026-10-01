"""Anchors for mcts_crisis --anchors-in from recorded games (eval --record-dir ... --record-tail N): for every game
that DIED, the recorded state K moves before its death (HISTORY 265: the strong actor's deaths are 60-120-move
slides from a healthy board, which the probe windows at 15/30 moves before death never search).

    python -m alphatrain.scripts.death_anchors --before 120 --out alphatrain/inference_cpp/data/slide120.txt \
        alphatrain/data/greedy_A5_cap100k alphatrain/data/greedy_A6_cap100k

One anchors.h line per death: seed turn cap b0..b80 r0 c0 col0 r1 c1 col1 r2 c2 col2 (unused balls: -1 -1 -1).
Run the miner with --prevention-turns K so each replay's label and spawn stream name the same offset.
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', type=int, required=True, help='moves before the death (<= the record tail)')
    ap.add_argument('--out', required=True)
    ap.add_argument('dirs', nargs='+')
    a = ap.parse_args()
    lines, seeds, balls, skipped = [], set(), [], 0
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, 'game_seed*.json'))):
            g = json.load(open(f))
            if not g.get('died'):
                continue
            if g.get('record_tail', 0) < a.before:
                raise SystemExit(f'{f}: record_tail {g.get("record_tail")} < --before {a.before}')
            want = g['final_turns'] - a.before
            st = next((s for s in g['states'] if s['turn'] == want), None)
            if st is None:          # the game died earlier than --before moves into the record
                skipped += 1
                continue
            if g['seed'] in seeds:
                raise SystemExit(f'duplicate seed {g["seed"]} across record dirs')
            seeds.add(g['seed'])
            board = np.array(st['board'], np.int64).reshape(81)
            nb = [(b['row'], b['col'], b['color']) for b in st['next_balls']][:3]
            nb += [(-1, -1, -1)] * (3 - len(nb))
            balls.append(int(np.count_nonzero(board)))
            lines.append(' '.join(str(x) for x in [g['seed'], want, want + 100000, *board.tolist(),
                                                     *[v for b in nb for v in b]]))
    with open(a.out, 'w') as fo:
        fo.write('\n'.join(lines) + '\n')
    print(f'wrote {len(lines)} anchors ({a.before} moves before death) to {a.out}; balls on board mean '
          f'{np.mean(balls):.1f} p10 {np.percentile(balls, 10):.0f} p90 {np.percentile(balls, 90):.0f}; '
          f'{skipped} deaths too early to rewind')


if __name__ == '__main__':
    main()
