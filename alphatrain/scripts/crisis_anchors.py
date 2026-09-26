"""Teacher-advantage check for an mcts_crisis run: does the search escape more deaths than the
greedy actor would from the SAME rewind states with the SAME spawn stream?

Each crisis replay starts from Game(seed) + SetState(anchor) (seed = original_seed*37 + rewind),
so a spawn-matched greedy control is `eval --anchors` on the same states and seeds.

  export:   python -m alphatrain.scripts.crisis_anchors export --crisis-dir data/gen1_crisis_A1 \
                --out alphatrain/inference_cpp/data/gen1_anchors.txt
  greedy:   (inference_cpp) ./build/eval --model data/A1_pair2_orig_e40_ts.pt --device mps \
                --batch 500 --anchors data/gen1_anchors.txt --scores-out data/gen1_anchors_greedy.csv
  compare:  python -m alphatrain.scripts.crisis_anchors compare --crisis-dir data/gen1_crisis_A1 \
                --greedy-csv alphatrain/inference_cpp/data/gen1_anchors_greedy.csv
"""
import argparse
import collections
import csv
import glob
import json
import os


def replays(crisis_dir):
    for f in sorted(glob.glob(os.path.join(crisis_dir, 'game_seed*.json'))):
        g = json.load(open(f))
        survived = g['turns'] - g['replay_from_turn']
        yield g, survived, survived >= g['continue_turns']


def export(a):
    n = 0
    with open(a.out, 'w') as out:
        for g, _, _ in replays(a.crisis_dir):
            m0 = g['moves'][0]
            cells = [str(v) for row in m0['board'] for v in row]
            assert len(cells) == 81
            nb = [f"{b['row']} {b['col']} {b['color']}" for b in m0['next_balls']]
            nb += ['-1 -1 -1'] * (3 - len(nb))
            out.write(f"{g['seed']} {g['replay_from_turn']} {g['continue_turns']} "
                      f"{' '.join(cells)} {' '.join(nb)}\n")
            n += 1
    print(f'wrote {n} anchors to {a.out}', flush=True)


def compare(a):
    greedy = {}
    for row in csv.DictReader(open(a.greedy_csv)):
        greedy[int(row['seed'])] = (int(row['turns']), row['capped'] == '1')
    table = collections.defaultdict(collections.Counter)
    surv = collections.defaultdict(lambda: ([], []))
    for g, s_survived, s_escaped in replays(a.crisis_dir):
        turns, g_escaped = greedy[g['seed']]
        kind = g['label']
        table[kind][(s_escaped, g_escaped)] += 1
        surv[kind][0].append(s_survived)
        surv[kind][1].append(turns - g['replay_from_turn'])
    for kind, t in sorted(table.items()):
        n = sum(t.values())
        s_rate = (t[(True, True)] + t[(True, False)]) / n
        g_rate = (t[(True, True)] + t[(False, True)]) / n
        print(f'{kind:10s} n={n}  escaped: search {s_rate:.1%}  greedy {g_rate:.1%}  '
              f'(both {t[(True, True)]}, search only {t[(True, False)]}, greedy only '
              f'{t[(False, True)]}, neither {t[(False, False)]})')
        for name, xs in zip(('search', 'greedy'), surv[kind]):
            xs = sorted(xs)
            print(f'    {name:6s} survived turns: P25 {xs[len(xs) // 4]}  median {xs[len(xs) // 2]}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export')
    e.add_argument('--crisis-dir', required=True)
    e.add_argument('--out', required=True)
    c = sub.add_parser('compare')
    c.add_argument('--crisis-dir', required=True)
    c.add_argument('--greedy-csv', required=True)
    a = ap.parse_args()
    export(a) if a.cmd == 'export' else compare(a)


if __name__ == '__main__':
    main()
