"""Compare mcts_relabel configurations on the same stored roots and export
per-config flip sets (visit argmax != prior argmax) for the rollout judge.

  .venv/bin/python alphatrain/scripts/relabel_compare.py \
      --states alphatrain/inference_cpp/data/relabel_pilot_states.bin \
      --configs vl1_s400 vmean_s400 ...
Writes alphatrain/inference_cpp/data/relabel_pilot_<config>_flips.bin per config.
"""
import argparse, csv, json, os, struct
import numpy as np


def load_bin(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'CLRJ'
        n = struct.unpack('<i', f.read(4))[0]
        recs = []
        for _ in range(n):
            board = f.read(81)
            nn = struct.unpack('<i', f.read(4))[0]
            nb = [struct.unpack('<iii', f.read(12)) for _ in range(3)]
            tm, bm, ts = struct.unpack('<iif', f.read(12))
            recs.append((board, nn, nb, tm, bm, ts))
    return recs


def write_bin(path, recs):
    with open(path, 'wb') as f:
        f.write(b'CLRJ'); f.write(struct.pack('<i', len(recs)))
        for board, nn, nb, tm, bm, ts in recs:
            f.write(board); f.write(struct.pack('<i', nn))
            for t in range(3): f.write(struct.pack('<iii', *nb[t]))
            f.write(struct.pack('<iif', tm, bm, ts))


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    return {
        'prior': np.array([int(r['prior_argmax']) for r in rows]),
        'visit': np.array([int(r['visit_argmax']) for r in rows]),
        'file_teacher': np.array([int(r['file_teacher']) for r in rows]),
        'top': np.array([float(r['top_share']) for r in rows]),
        'rows': rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--states', default='alphatrain/inference_cpp/data/relabel_pilot_states.bin')
    p.add_argument('--configs', nargs='+', required=True)
    p.add_argument('--prefix', default='alphatrain/inference_cpp/data/relabel_pilot_')
    p.add_argument('--base-config', default=None,
                   help='take the BASE move (prior argmax) from this config instead of each config\'s own prior '
                        '(use when the teacher is a different model, e.g. pillar3k --sims 1)')
    a = p.parse_args()
    recs = load_bin(a.states)
    meta = json.load(open(a.states + '.meta.json'))
    strat = np.array([m['stratum'] for m in meta])
    tail = strat == 'tail'
    data = {c: load_csv(f'{a.prefix}{c}.csv') for c in a.configs}
    if a.base_config:
        base = load_csv(f'{a.prefix}{a.base_config}.csv')['prior']
        for d in data.values():
            d['prior'] = base
    n = len(recs)
    print(f'{n} states: tail={tail.sum()} broad={(~tail).sum()}')
    print(f'{"config":22s} {"flip%":>6s} {"tail%":>6s} {"broad%":>6s} {"=file%":>6s} '
          f'{"top_all":>7s} {"top_tail":>8s}')
    ref = data[a.configs[0]]
    for c, d in data.items():
        flip = d['visit'] != d['prior']
        print(f'{c:22s} {100*flip.mean():6.2f} {100*flip[tail].mean():6.2f} '
              f'{100*flip[~tail].mean():6.2f} {100*(d["visit"]==d["file_teacher"]).mean():6.2f} '
              f'{d["top"].mean():7.3f} {d["top"][tail].mean():8.3f}')
        # export flip set for the judge: teacher = new visit argmax, base = prior argmax
        out = []
        for i in np.where(flip)[0]:
            board, nn, nb, _, _, _ = recs[i]
            out.append((board, nn, nb, int(d['visit'][i]), int(d['prior'][i]), float(d['top'][i])))
        write_bin(f'{a.prefix}{c}_flips.bin', out)
        json.dump([int(i) for i in np.where(flip)[0]], open(f'{a.prefix}{c}_flips.idx.json', 'w'))
    print('\npairwise: states where both flip vs prior / same target among those / '
          'visit-argmax agreement overall')
    cs = a.configs
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            A, B = data[cs[i]], data[cs[j]]
            fa, fb = A['visit'] != A['prior'], B['visit'] != B['prior']
            both = fa & fb
            same = (A['visit'] == B['visit'])[both].mean() if both.any() else float('nan')
            print(f'  {cs[i]:20s} vs {cs[j]:20s}: both {both.sum():4d}  same-target '
                  f'{100*same:5.1f}%  agree-all {100*(A["visit"]==B["visit"]).mean():5.2f}%')


if __name__ == '__main__':
    main()
