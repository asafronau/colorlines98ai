"""Path B v2 — Phase 1: generate B_ep12 policy-only argmax self-play games.

Plays N games using B_ep12's policy (argmax over legal moves). Saves a
per-turn trajectory for each game so the downstream stratified sampler can
extract crisis / uncertain / diverse anchor states.

Uses the fleet_jit batched step pattern: M slots in parallel, one batched B
forward per step, fleet_jit handles legal argmax + line clears + spawn.
Slots refill from a seed queue as games complete.

Output is a single .pt file:
{
  'args': {...},
  'games': [
    {
      'seed': int,
      'final_score': int,
      'final_turns': int,
      'died': bool,
      'capped': bool,
      'boards':   int8  [T, 9, 9],
      'next_pos': int8  [T, 3, 2],
      'next_col': int8  [T, 3],
      'n_next':   int8  [T],
      'scores':   int32 [T],
    },
    ...
  ]
}

T is the number of states recorded (subject to --save-every).

Usage:
    python -m alphatrain.scripts.gen_b_selfplay \\
        --checkpoint alphatrain/data/b_smoke_epoch_12.pt \\
        --output alphatrain/data/b_selfplay.pt \\
        --n-games 500 --seed-start 100000 \\
        --fleet-size 64 --max-turns 8000 --device mps
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from alphatrain.model import AlphaTrainNet
from alphatrain.scripts.fleet_jit import build_obs_fleet_jit, step_fleet_jit
from game.board import ColorLinesGame


def load_model(checkpoint_path, device, num_blocks=10, channels=256):
    ckpt = torch.load(checkpoint_path, map_location=device,
                       weights_only=False)
    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model = AlphaTrainNet(num_blocks=num_blocks,
                          channels=channels).to(device)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                 if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.train(False)
    return model, ckpt.get('epoch', '?')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--n-games', type=int, default=500)
    p.add_argument('--seed-start', type=int, default=100_000)
    p.add_argument('--fleet-size', type=int, default=64)
    p.add_argument('--max-turns', type=int, default=8000)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--device', default='mps')
    p.add_argument('--save-every', type=int, default=1,
                   help='Record every Nth turn (1 = every turn).')
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    print(f"\nLoading {args.checkpoint}...", flush=True)
    model, epoch = load_model(args.checkpoint, device,
                                args.num_blocks, args.channels)
    print(f"  epoch={epoch}", flush=True)

    M = args.fleet_size
    s_boards = np.zeros((M, 9, 9), dtype=np.int8)
    s_next_pos = np.zeros((M, 3, 2), dtype=np.int8)
    s_next_col = np.zeros((M, 3), dtype=np.int8)
    s_n_next = np.zeros(M, dtype=np.int8)
    s_scores = np.zeros(M, dtype=np.int32)
    s_turns = np.zeros(M, dtype=np.int32)
    s_game_overs = np.zeros(M, dtype=np.bool_)
    s_rng_states = np.zeros(M, dtype=np.uint64)
    s_seed_for_slot = np.full(M, -1, dtype=np.int64)
    s_active = np.zeros(M, dtype=np.bool_)
    # Per-slot trajectory buffers (lists of state records appended each step)
    s_traj_boards = [[] for _ in range(M)]
    s_traj_next_pos = [[] for _ in range(M)]
    s_traj_next_col = [[] for _ in range(M)]
    s_traj_n_next = [[] for _ in range(M)]
    s_traj_scores = [[] for _ in range(M)]

    queue = list(range(args.seed_start, args.seed_start + args.n_games))
    queue_idx = 0

    def load_slot(slot, seed):
        g = ColorLinesGame(seed=seed)
        g.reset()
        s_boards[slot] = g.board.astype(np.int8)
        for k in range(3):
            if k < len(g.next_balls):
                pos, col = g.next_balls[k]
                s_next_pos[slot, k, 0] = int(pos[0])
                s_next_pos[slot, k, 1] = int(pos[1])
                s_next_col[slot, k] = int(col)
            else:
                s_next_pos[slot, k, 0] = 0
                s_next_pos[slot, k, 1] = 0
                s_next_col[slot, k] = 0
        s_n_next[slot] = min(len(g.next_balls), 3)
        s_scores[slot] = 0
        s_turns[slot] = 0
        s_game_overs[slot] = False
        # Derive an RNG state from seed; matches phase1_oracle_fleet style.
        s_rng_states[slot] = np.uint64(seed * 7919 + 1)
        s_seed_for_slot[slot] = seed
        s_active[slot] = True
        s_traj_boards[slot] = []
        s_traj_next_pos[slot] = []
        s_traj_next_col[slot] = []
        s_traj_n_next[slot] = []
        s_traj_scores[slot] = []

    for slot in range(M):
        if queue_idx < len(queue):
            load_slot(slot, queue[queue_idx])
            queue_idx += 1

    finished_games = []
    t0 = time.time()
    n_steps = 0
    last_log = t0

    @torch.no_grad()
    def batched_forward(obs_np):
        ob_t = torch.from_numpy(obs_np).to(device)
        out = model(ob_t)
        if isinstance(out, tuple):
            out = out[0]
        return out.float().cpu().numpy()

    while s_active.any():
        active_idx = np.where(s_active)[0]
        n_active = len(active_idx)

        active_boards = s_boards[active_idx]
        active_next_pos = s_next_pos[active_idx]
        active_next_col = s_next_col[active_idx]
        active_n_next = s_n_next[active_idx]
        active_turns = s_turns[active_idx].copy()
        active_scores = s_scores[active_idx].copy()
        active_game_overs = s_game_overs[active_idx].copy()
        active_completion = np.zeros(n_active, dtype=np.int8)
        active_rng = s_rng_states[active_idx].copy()

        # Build obs
        obs_active = np.empty((n_active, 18, 9, 9), dtype=np.float32)
        build_obs_fleet_jit(active_boards, active_next_pos,
                             active_next_col, active_n_next, obs_active)

        # Forward B (logits, NOT probs — step_fleet_jit just needs argmax)
        logits = batched_forward(obs_active).astype(np.float32)

        # Save snapshot BEFORE step for active slots that match save_every
        for ai, slot in enumerate(active_idx):
            if active_turns[ai] % args.save_every == 0:
                s_traj_boards[slot].append(active_boards[ai].copy())
                s_traj_next_pos[slot].append(active_next_pos[ai].copy())
                s_traj_next_col[slot].append(active_next_col[ai].copy())
                s_traj_n_next[slot].append(int(active_n_next[ai]))
                s_traj_scores[slot].append(int(active_scores[ai]))

        # Step (argmax legal + line clear + spawn)
        step_fleet_jit(active_boards, active_next_pos, active_next_col,
                        active_n_next, active_scores, active_turns,
                        active_game_overs, active_completion,
                        logits, active_rng)

        # Write back per-slot state from the (possibly mutated) active copies
        s_boards[active_idx] = active_boards
        s_next_pos[active_idx] = active_next_pos
        s_next_col[active_idx] = active_next_col
        s_n_next[active_idx] = active_n_next
        s_scores[active_idx] = active_scores
        s_turns[active_idx] = active_turns
        s_game_overs[active_idx] = active_game_overs
        s_rng_states[active_idx] = active_rng

        # Handle completions
        for ai, slot in enumerate(active_idx):
            died = (active_completion[ai] == 1) or s_game_overs[slot]
            capped = (not died) and (s_turns[slot] >= args.max_turns)
            if died or capped:
                T = len(s_traj_boards[slot])
                game = {
                    'seed': int(s_seed_for_slot[slot]),
                    'final_score': int(s_scores[slot]),
                    'final_turns': int(s_turns[slot]),
                    'died': bool(died),
                    'capped': bool(capped),
                    'boards': np.stack(s_traj_boards[slot]).astype(np.int8)
                                if T else np.zeros((0, 9, 9), dtype=np.int8),
                    'next_pos': np.stack(s_traj_next_pos[slot]).astype(np.int8)
                                  if T else np.zeros((0, 3, 2), dtype=np.int8),
                    'next_col': np.stack(s_traj_next_col[slot]).astype(np.int8)
                                  if T else np.zeros((0, 3), dtype=np.int8),
                    'n_next': np.array(s_traj_n_next[slot], dtype=np.int8),
                    'scores': np.array(s_traj_scores[slot], dtype=np.int32),
                }
                finished_games.append(game)
                if queue_idx < len(queue):
                    load_slot(slot, queue[queue_idx])
                    queue_idx += 1
                else:
                    s_active[slot] = False

        n_steps += 1
        now = time.time()
        if now - last_log > 15.0:
            elapsed = now - t0
            n_done = len(finished_games)
            rate = n_done / max(elapsed, 1e-3)
            remaining = len(queue) - n_done
            eta = remaining / max(rate, 1e-3)
            avg_turns = np.mean([g['final_turns']
                                  for g in finished_games[-50:]]) \
                          if finished_games else 0
            avg_score = np.mean([g['final_score']
                                  for g in finished_games[-50:]]) \
                          if finished_games else 0
            print(f"  step {n_steps} | active={n_active} | "
                  f"done={n_done}/{len(queue)} | "
                  f"recent avg score={avg_score:.0f} turns={avg_turns:.0f} | "
                  f"rate={rate:.2f} g/s | "
                  f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s", flush=True)
            last_log = now

    print(f"\nDone: {len(finished_games)} games in {time.time()-t0:.0f}s",
          flush=True)
    print(f"  mean score: {np.mean([g['final_score'] for g in finished_games]):.0f}",
          flush=True)
    print(f"  mean turns: {np.mean([g['final_turns'] for g in finished_games]):.0f}",
          flush=True)
    print(f"  died: {sum(g['died'] for g in finished_games)}, "
          f"capped: {sum(g['capped'] for g in finished_games)}", flush=True)

    out = {'args': vars(args), 'games': finished_games}
    torch.save(out, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved {args.output} ({size_mb:.0f} MB)", flush=True)


if __name__ == '__main__':
    main()
