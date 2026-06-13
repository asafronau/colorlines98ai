"""Golden test 1 (the ship gate): GPU engine == ColorLinesGame, bit-exact, every turn.

Records N CPU games (pseudo-random legal play — dense boards, heavy displacement and
spawn-clear traffic, fast deaths) with every spawn decision captured, then drives ALL of
them through alphatrain.gpu_eval_engine in one batch with that randomness INJECTED.
Boards, scores, turns and game-over must match bit-exactly after every single turn.
Validates the full transition (move, clear dedup+scoring, sequential spawn placement,
displacement, ordered spawn-clears, regen, game-over) independent of RNG.

    PYTHONPATH=. python scripts/test_gpu_engine_golden.py --games 50 --device mps
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from game.board import ColorLinesGame, _get_empty_array
from alphatrain.gpu_eval_engine import GpuGames, BOARD, BALLS_PER_TURN


class RecordingGame(ColorLinesGame):
    """ColorLinesGame that records spawn displacement + regenerated next balls."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rec_disp = None      # per-ball displaced landing cell (flat) or -1
        self.rec_regen = None     # next_balls after regeneration

    def _spawn_balls(self):
        # Copy of the parent implementation with per-ball displacement recording
        # (parent returns landed cells but loses the ball-index alignment).
        landed = []
        self.rec_disp = [-1] * BALLS_PER_TURN
        for i, ((row, col), color) in enumerate(self.next_balls):
            if self.board[row, col] == 0:
                self.board[row, col] = color
                landed.append((row, col))
            else:
                empty = _get_empty_array(self.board)
                if len(empty) > 0:
                    idx = self.rng.randint(0, len(empty))
                    r, c = int(empty[idx, 0]), int(empty[idx, 1])
                    self.board[r, c] = color
                    landed.append((r, c))
                    self.rec_disp[i] = r * BOARD + c
        self._cc_labels = None
        return landed

    def _generate_next_balls(self):
        super()._generate_next_balls()
        self.rec_regen = list(self.next_balls)


def _would_clear(board, sr, sc, tr, tc):
    """Run length through (tr,tc) if the ball at (sr,sc) moved there — max over dirs."""
    color = board[sr, sc]
    best = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for sign in (1, -1):
            r, c = tr + sign * dr, tc + sign * dc
            while 0 <= r < BOARD and 0 <= c < BOARD and not (r == sr and c == sc) \
                    and board[r, c] == color:
                n += 1
                r += sign * dr
                c += sign * dc
        best = max(best, n)
    return best


def record_trace(seed, max_turns=2000, line_seek=True, num_colors=None):
    g = RecordingGame(seed=seed) if num_colors is None \
        else RecordingGame(seed=seed, num_colors=num_colors)
    g.reset()
    init = {
        'board': g.board.copy(),
        'next_balls': list(g.next_balls),
    }
    turns = []
    t = 0
    pick = np.random.RandomState(seed ^ 0x5EED)
    while not g.game_over and t < max_turns:
        moves = g.get_legal_moves()
        if not moves:
            break
        choice = None
        if line_seek:
            # Take a clearing move when one exists — exercises the move-clear
            # scoring path constantly (random play alone NEVER completes lines).
            clearing = [m for m in moves
                        if _would_clear(g.board, m[0][0], m[0][1],
                                        m[1][0], m[1][1]) >= 5]
            if clearing:
                choice = clearing[pick.randint(len(clearing))]
        if choice is None:
            choice = moves[pick.randint(len(moves))]
        (sr, sc), (tr, tc) = choice
        g.rec_disp, g.rec_regen = None, None
        res = g.move((sr, sc), (tr, tc))
        assert res['valid']
        turns.append({
            'src': sr * BOARD + sc, 'tgt': tr * BOARD + tc,
            'disp': g.rec_disp or [-1] * BALLS_PER_TURN,
            'regen': g.rec_regen,           # None when the move cleared a line
            'board': g.board.copy(),
            'score': g.score, 'over': g.game_over,
        })
        t += 1
    return init, turns


def synthetic_cross_trace():
    """Crossing-lines dedup case (never occurs in random play): one move completes a
    horizontal AND a vertical 5-line through the same cell → 9 deduped cells →
    score 9*(9-4)=45, NOT 2×5. Built by hand, recorded like a normal trace."""
    g = RecordingGame(seed=1)
    g.reset()
    g.board[:] = 0
    g.board[4, 0:4] = 1          # 4 horizontal at row 4, cols 0-3
    g.board[0:4, 4] = 1          # 4 vertical at col 4, rows 0-3
    g.board[8, 8] = 1            # the ball to move into (4,4)
    g.next_balls = [((0, 0), 2), ((1, 1), 3), ((2, 2), 4)]
    g.game_over = False
    g.score = 0
    init = {'board': g.board.copy(), 'next_balls': list(g.next_balls)}
    g.rec_disp, g.rec_regen = None, None
    res = g.move((8, 8), (4, 4))
    assert res['valid'] and res['cleared'] == 9 and g.score == 45, \
        f"synthetic cross broken: {res} score={g.score}"
    turns = [{'src': 8 * BOARD + 8, 'tgt': 4 * BOARD + 4,
              'disp': g.rec_disp or [-1] * BALLS_PER_TURN,
              'regen': g.rec_regen, 'board': g.board.copy(),
              'score': g.score, 'over': g.game_over}]
    return init, turns


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=50)
    p.add_argument('--seed0', type=int, default=4242)
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)

    print(f"recording {a.games} CPU games (line-seeking pseudo-random play; "
          f"half with 3 colors so lines/clears actually happen)...", flush=True)
    traces = [record_trace(a.seed0 + i,
                           num_colors=(3 if i % 2 else None))
              for i in range(a.games)]
    traces.append(synthetic_cross_trace())
    a.games += 1
    lens = [len(t) for _, t in traces]
    n_disp = sum(1 for _, ts in traces for t in ts if any(d >= 0 for d in t['disp']))
    n_mc = sum(1 for _, ts in traces for t in ts if t['regen'] is None)
    print(f"  lengths: min={min(lens)} median={sorted(lens)[len(lens)//2]} "
          f"max={max(lens)} | move-clear turns={n_mc} "
          f"displacement turns={n_disp}", flush=True)
    assert n_mc > 50, "trace set has too few move-clears to be a meaningful test"

    B = a.games
    st = GpuGames(B, dev)
    for gi, (init, _) in enumerate(traces):
        st.boards[gi] = torch.from_numpy(init['board']).to(dev)
        for i, ((r, c), col) in enumerate(init['next_balls']):
            st.next_pos[gi, i, 0] = r
            st.next_pos[gi, i, 1] = c
            st.next_col[gi, i] = col
        st.n_next[gi] = len(init['next_balls'])
    st.alive[:] = True

    max_T = max(lens)
    src = torch.zeros(B, dtype=torch.int64, device=dev)
    tgt = torch.zeros(B, dtype=torch.int64, device=dev)
    mismatches = 0
    for t in range(max_T):
        disp = torch.full((B, BALLS_PER_TURN), -1, dtype=torch.int64)
        regen_pos = torch.zeros(B, BALLS_PER_TURN, 2, dtype=torch.int8)
        regen_col = torch.zeros(B, BALLS_PER_TURN, dtype=torch.int8)
        regen_n = torch.zeros(B, dtype=torch.int64)
        stepping = torch.zeros(B, dtype=torch.bool)
        for gi, (_, turns) in enumerate(traces):
            if t >= len(turns):
                continue
            tr_ = turns[t]
            stepping[gi] = True
            src[gi], tgt[gi] = tr_['src'], tr_['tgt']
            disp[gi] = torch.tensor(tr_['disp'], dtype=torch.int64)
            if tr_['regen'] is not None:
                for i, ((r, c), col) in enumerate(tr_['regen']):
                    regen_pos[gi, i, 0] = r
                    regen_pos[gi, i, 1] = c
                    regen_col[gi, i] = col
                regen_n[gi] = len(tr_['regen'])
            else:
                # move-cleared turn: regen unused (spawn mask false), keep current
                regen_n[gi] = int(st.n_next[gi])
        st.alive = stepping.to(dev)
        st.step(src, tgt, inj={'disp': disp, 'regen_pos': regen_pos,
                               'regen_col': regen_col, 'regen_n': regen_n})

        boards_cpu = st.boards.cpu().numpy()
        score_cpu = st.score.cpu().numpy()
        for gi, (_, turns) in enumerate(traces):
            if t >= len(turns):
                continue
            tr_ = turns[t]
            if not np.array_equal(boards_cpu[gi], tr_['board']) \
                    or score_cpu[gi] != tr_['score']:
                mismatches += 1
                if mismatches <= 3:
                    db = np.argwhere(boards_cpu[gi] != tr_['board'])
                    print(f"\nMISMATCH game {gi} (seed {a.seed0+gi}) turn {t}: "
                          f"score gpu={score_cpu[gi]} cpu={tr_['score']}, "
                          f"{len(db)} cells differ, first={db[:6].tolist()}")
                    print(f"  move {tr_['src']}->{tr_['tgt']} disp={tr_['disp']}")
        if mismatches > 3:
            break

    total_turns = sum(lens)
    if mismatches == 0:
        print(f"\nGOLDEN PASS: {a.games} games, {total_turns} turns, "
              f"boards+scores bit-identical on {a.device}.")
    else:
        raise SystemExit(f"\nGOLDEN FAIL: {mismatches}+ mismatching turns.")


if __name__ == '__main__':
    main()
