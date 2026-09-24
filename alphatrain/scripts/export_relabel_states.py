"""Export stored MCTS roots from full-record selfplay games to the CLRJ state
format read by mcts_relabel / rollout_judge.

Strata (per game):
  tail  -- every root in the last --tail moves of a game that DIED (not capped)
  broad -- --broad random roots per game, from all games
teacher_move = recorded chosen move (visit argmax), base_move = prior argmax,
top_share = recorded max_visits / sum_visits.  A sidecar JSON carries per-state
metadata (seed, turn, stratum, capped, turns_to_end) in file order.
"""
import argparse, glob, json, os, struct, sys
import numpy as np


def flat(m):
    return (m['sr'] * 9 + m['sc']) * 81 + (m['tr'] * 9 + m['tc'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', default='alphatrain/data/flywheel_18b96e40_i1_pilot/exploit')
    p.add_argument('--out', default='alphatrain/inference_cpp/data/relabel_pilot_states.bin')
    p.add_argument('--tail', type=int, default=20)
    p.add_argument('--broad', type=int, default=1, help='random roots per game')
    p.add_argument('--max-games', type=int, default=0)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    files = sorted(glob.glob(os.path.join(a.games_dir, 'game_seed*.json')))
    if a.max_games:
        files = files[:a.max_games]
    recs, meta = [], []
    n_dead = 0
    for fi, f in enumerate(files):
        d = json.load(open(f))
        moves = d['moves']
        T = len(moves)
        dead = not d['capped']
        n_dead += dead
        picks = []
        if dead:
            picks += [(t, 'tail') for t in range(max(0, T - a.tail), T)]
        for t in rng.choice(T, min(a.broad, T), replace=False):
            picks.append((int(t), 'broad'))
        for t, stratum in picks:
            m = moves[t]
            board = np.array(m['board'], dtype=np.int8).reshape(81)
            nb = m['next_balls']
            vis = np.array(m['cand_visits'], dtype=np.float64)
            pri = np.array(m['cand_prior'], dtype=np.float64)
            acts = m['cand_moves']
            teacher = flat(m['chosen_move'])
            base = int(acts[int(pri.argmax())])
            top = float(vis.max() / vis.sum()) if vis.sum() > 0 else 0.0
            recs.append((board, nb, teacher, base, top))
            meta.append({'seed': d['seed'], 'turn': t, 'stratum': stratum,
                         'capped': d['capped'], 'turns_to_end': T - t,
                         'file_teacher': teacher, 'file_base': base, 'top_share': top})
        if (fi + 1) % 500 == 0:
            print(f'  {fi+1}/{len(files)} games, {len(recs)} states', flush=True)
    with open(a.out, 'wb') as f:
        f.write(b'CLRJ')
        f.write(struct.pack('<i', len(recs)))
        for board, nb, teacher, base, top in recs:
            f.write(board.tobytes())
            f.write(struct.pack('<i', len(nb)))
            for t in range(3):
                if t < len(nb):
                    f.write(struct.pack('<iii', nb[t]['row'], nb[t]['col'], nb[t]['color']))
                else:
                    f.write(struct.pack('<iii', 0, 0, 0))
            f.write(struct.pack('<iif', teacher, base, top))
    json.dump(meta, open(a.out + '.meta.json', 'w'))
    st = [m['stratum'] for m in meta]
    flips = sum(m['file_teacher'] != m['file_base'] for m in meta)
    print(f'wrote {a.out}: {len(recs)} states from {len(files)} games ({n_dead} died); '
          f'tail={st.count("tail")} broad={st.count("broad")}; recorded teacher!=prior: '
          f'{flips} ({100*flips/len(recs):.2f}%)')


if __name__ == '__main__':
    main()
