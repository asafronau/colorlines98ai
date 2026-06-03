"""Batched death-game recorder — pass 1 of mining.

Plays B games to natural death IN PARALLEL (one batched forward per step, slot
refill on death), recording each one's last `tail` frames into a death_<seed>.json
that mine_crisis_sweep consumes directly. Replaces overnight's serial, batch-1,
one-game-at-a-time recording (the GPU sat idle there). Per frame we store only
what mining needs: board, next_balls, score_before, turn, empties, chosen_move.

Game-play matches find_worst_game's greedy-argmax-legal exactly (same
_build_obs_for_game + _get_legal_priors_flat + game.move); in fp32 it's
batch-invariant, so a batched recording reproduces the single-game one bit-for-bit
(golden-tested). In fp16 batched it produces equally-valid games (fp16 batch noise
just yields different — still real — crisis positions).

    PYTHONPATH=. python scripts/batch_record.py --seed-start 90000 --n-seeds 200 \\
        --device mps --batch 128
"""
import os
import sys
import json
import time
import argparse
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.evaluate import load_model
from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
from scripts.batched_rollout import _decode


class _Slot:
    __slots__ = ('seed', 'game', 'frames')

    def __init__(self, seed, tail):
        self.seed = seed
        self.game = ColorLinesGame(seed=seed)
        self.game.reset()
        self.frames = collections.deque(maxlen=tail)


def _frame(game, chosen):
    board = game.board
    return {
        'turn': int(game.turns),
        'score_before': int(game.score),
        'score': int(game.score),
        'board': board.astype(np.int8).tolist(),
        'next_balls': [[[int(p[0]), int(p[1])], int(c)] for p, c in game.next_balls],
        'empties': int((board == 0).sum()),
        'chosen_move': [list(chosen[0]), list(chosen[1])],
    }


def _save(slot, out_dir, model, max_turns):
    d = {'seed': slot.seed, 'model': model,
         'final_score': int(slot.game.score), 'final_turns': int(slot.game.turns),
         'died': bool(slot.game.game_over), 'max_turns': max_turns,
         'frames': list(slot.frames)}
    json.dump(d, open(os.path.join(out_dir, f'death_{slot.seed}.json'), 'w'))
    return d['died'], d['final_turns']


def batch_record(net, dev, dtype, seeds, out_dir, model, batch=128, tail=60,
                 max_turns=1_000_000, skip_existing=True, log_every=50):
    os.makedirs(out_dir, exist_ok=True)
    todo = [s for s in seeds if not (skip_existing and
            os.path.exists(os.path.join(out_dir, f'death_{s}.json')))]
    nxt = 0

    def make_slot():
        nonlocal nxt
        if nxt < len(todo):
            s = todo[nxt]; nxt += 1
            return _Slot(s, tail)
        return None

    slots = [s for s in (make_slot() for _ in range(batch)) if s is not None]
    done = 0
    t0 = time.perf_counter()
    while slots:
        obs = np.stack([_build_obs_for_game(s.game) for s in slots])
        with torch.no_grad():
            logits = net(torch.from_numpy(obs).to(dev, dtype)).float().cpu().numpy()
        survivors, finalized = [], 0
        for i, s in enumerate(slots):
            priors = _get_legal_priors_flat(s.game.board, logits[i], 30)
            if not priors:                       # no legal move = stuck/dead
                s.game.game_over = True
                _save(s, out_dir, model, max_turns); finalized += 1; done += 1
                continue
            chosen = _decode(max(priors.items(), key=lambda x: x[1])[0])
            s.frames.append(_frame(s.game, chosen))
            if not s.game.move(*chosen)['valid']:
                _save(s, out_dir, model, max_turns); finalized += 1; done += 1
                continue
            if s.game.game_over or s.game.turns >= max_turns:
                _save(s, out_dir, model, max_turns); finalized += 1; done += 1
                if done % log_every == 0:
                    print(f"  recorded {done}/{len(todo)} "
                          f"({done/(time.perf_counter()-t0):.1f} games/s)", flush=True)
            else:
                survivors.append(s)
        for _ in range(finalized):
            ns = make_slot()
            if ns is not None:
                survivors.append(ns)
        slots = survivors
    dt = time.perf_counter() - t0
    print(f"batch_record: {done} games in {dt:.0f}s ({done/max(dt,1e-9):.2f} games/s) "
          f"-> {out_dir}", flush=True)
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--seed-start', type=int, required=True)
    p.add_argument('--n-seeds', type=int, required=True)
    p.add_argument('--out-dir', default='alphatrain/data/death_games')
    p.add_argument('--device', default='mps')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--tail', type=int, default=60)
    p.add_argument('--max-turns', type=int, default=1_000_000)
    p.add_argument('--fp32', action='store_true')
    a = p.parse_args()
    dev = torch.device(a.device)
    fp16 = (not a.fp32) and dev.type != 'cpu'
    net, _ = load_model(a.model, dev, fp16=fp16)
    dtype = next(net.parameters()).dtype
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    print(f"batch_record: seeds {a.seed_start}..{a.seed_start+a.n_seeds-1}, "
          f"device={a.device} dtype={dtype} batch={a.batch} tail={a.tail}", flush=True)
    batch_record(net, dev, dtype, seeds, a.out_dir, a.model, batch=a.batch,
                 tail=a.tail, max_turns=a.max_turns)


if __name__ == '__main__':
    main()
