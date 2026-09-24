"""Export the final --tail states of recorded greedy games (eval --record-dir) to CLRJ for
mcts_relabel. teacher_move = base_move = the student's greedy move (placeholder; the relabel
supplies the teacher). Sidecar meta: seed, turn, turns_to_end, final_turns."""
import argparse, glob, json, os, struct
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tail', type=int, default=200)
    p.add_argument('--every', type=int, default=0, help='also keep every k-th earlier state (0=off)')
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.games_dir, 'game_seed*.json')))
    recs, meta = [], []
    for fi, f in enumerate(files):
        g = json.load(open(f)); st = g['states']; T = g['final_turns']
        keep = [s for s in st if s['turn'] >= T - a.tail]
        if a.every: keep += [s for s in st if s['turn'] < T - a.tail and s['turn'] % a.every == 0]
        for s in keep:
            recs.append((np.array(s['board'], np.int8).reshape(81), s['next_balls'], int(s['move'])))
            meta.append({'seed': g['seed'], 'turn': s['turn'], 'turns_to_end': T - s['turn'], 'final_turns': T, 'died': g['died']})
        if (fi + 1) % 200 == 0: print(f'  {fi+1}/{len(files)} games, {len(recs)} states', flush=True)
    with open(a.out, 'wb') as f:
        f.write(b'CLRJ'); f.write(struct.pack('<i', len(recs)))
        for b, nb, mv in recs:
            f.write(b.tobytes()); f.write(struct.pack('<i', len(nb)))
            for t in range(3):
                f.write(struct.pack('<iii', nb[t]['row'], nb[t]['col'], nb[t]['color']) if t < len(nb) else struct.pack('<iii', 0, 0, 0))
            f.write(struct.pack('<iif', mv, mv, 0.0))
    json.dump(meta, open(a.out + '.meta.json', 'w'))
    print(f'wrote {a.out}: {len(recs)} states from {len(files)} games')


if __name__ == '__main__':
    main()
