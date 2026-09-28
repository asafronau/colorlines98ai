"""Escape rates from troubled states (HISTORY 253-254): one anchors file (anchors.h format, written by
mcts_crisis --anchors-out or crisis_anchors.py export) and the per-anchor CSVs that the C++ engine wrote
for it (`eval --anchors ... --scores-out X.csv`). Escaped = reached the anchor's turn cap (the
anchors file's cap field; 200 moves from 2026-09-28). Rewind depth = seed % 37 (15 or 30 moves
before the probe player's death).

    python -m alphatrain.scripts.escape_benchmark --anchors alphatrain/inference_cpp/data/bench.txt \
        A1=alphatrain/inference_cpp/data/bench_A1.csv A3=alphatrain/inference_cpp/data/bench_A3.csv
"""
import argparse
import collections
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchors', required=True)
    ap.add_argument('runs', nargs='+', help='name=csv')
    a = ap.parse_args()
    depth = {}
    for line in open(a.anchors):
        seed = int(line.split(None, 1)[0])
        depth[seed] = seed % 37
    print(f'{len(depth)} anchors, rewind depths {dict(sorted(collections.Counter(depth.values()).items()))}')
    for run in a.runs:
        name, path = run.split('=', 1)
        esc, n = collections.Counter(), collections.Counter()
        for r in csv.DictReader(open(path)):
            d = depth[int(r['seed'])]
            n[d] += 1
            esc[d] += r['capped'] == '1'
        missing = len(depth) - sum(n.values())
        if missing:
            raise SystemExit(f'{name}: {missing} anchors missing from {path}')
        cells = '  '.join(f'{d} before death: {esc[d] / n[d]:6.1%}' for d in sorted(n, reverse=True))
        print(f'{name:28s} {cells}   overall {sum(esc.values()) / sum(n.values()):6.1%}')


if __name__ == '__main__':
    main()
