"""Path B v2 — Phase 3: K-rollout mining on B_ep12-distribution anchors.

Reads `sample_b_anchors.py` output and runs K common-RNG rollouts per
candidate move per anchor. Designed to be used twice:

  1. Stage 1 / calibration: --k 24, on a small subset (--max-anchors 100
     for calibration, or full set for cheap screen).
  2. Stage 2: --k 128, on screen-pass anchors.

Output is per (anchor, move) survival and turn statistics, plus
split-half aggregations so the downstream filter can apply stability
gates without re-running rollouts.

This is a focused rewrite of phase1_oracle_fleet.py's orchestration:
loader takes .pt instead of JSON dirs; the fleet step is identical;
aggregation adds split-half cap_rates.

Output schema:
{
  'args': {...},
  'anchors': [
    {
      'anchor_idx': int,
      'anchor_input_meta': {seed_origin, turn_origin, source_label,
                              turns_to_death, final_score, ...},
      'board': int8 (9, 9),
      'next_balls': [((r, c), col), ...],
      'top_moves': [{
        'rank': 1-based,
        'move_int': int,
        'prior': float,
        'n_rollouts': int,
        'cap_rate': float,
        'cap_rate_half1': float,
        'cap_rate_half2': float,
        'mean_turns': float,
        'mean_score': float,
      }, ...],
    },
    ...
  ],
  'summary': {
    'n_anchors_input': int,
    'n_anchors_kept': int,
    'n_rollouts': int,
  },
}

Usage (calibration):
    python -m alphatrain.scripts.mine_b_anchors \\
        --model alphatrain/data/b_smoke_epoch_12.pt \\
        --anchor-pt alphatrain/data/b_anchors.pt \\
        --max-anchors 100 \\
        --k 128 --horizon 300 --top-moves 6 \\
        --fleet-size 512 --device mps \\
        --output alphatrain/data/b_anchors_k128_calib.pt

Usage (Stage 1 screen):
    python -m alphatrain.scripts.mine_b_anchors \\
        --model alphatrain/data/b_smoke_epoch_12.pt \\
        --anchor-pt alphatrain/data/b_anchors.pt \\
        --k 24 --horizon 300 --top-moves 6 \\
        --output alphatrain/data/b_anchors_k24_screen.pt

Usage (Stage 2 K=128 on screen-pass):
    python -m alphatrain.scripts.mine_b_anchors \\
        --model alphatrain/data/b_smoke_epoch_12.pt \\
        --anchor-pt alphatrain/data/b_anchors_screen_pass.pt \\
        --k 128 --horizon 300 --top-moves 6 \\
        --output alphatrain/data/b_anchors_k128_full.pt
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from alphatrain.mcts import _get_legal_priors_flat
from alphatrain.model import AlphaTrainNet
from alphatrain.observation import build_observation
from alphatrain.scripts.fleet_jit import build_obs_fleet_jit, step_fleet_jit
from game.board import ColorLinesGame


def load_b_model(checkpoint_path, device, num_blocks=10, channels=256):
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


def realize_afterstate(board, next_balls, turn_origin, seed, move):
    """Apply a flat-int move to the anchor state; return afterstate dict
    or None if the move was invalid."""
    g = ColorLinesGame(seed=seed)
    g.reset(board=board.copy(), next_balls=list(next_balls))
    g.turns = turn_origin
    sr = move // 81 // 9
    sc = move // 81 % 9
    tr = move % 81 // 9
    tc = move % 81 % 9
    r = g.move((sr, sc), (tr, tc))
    if not r['valid']:
        return None
    return {
        'board': g.board.copy(),
        'next_balls': list(g.next_balls),
        'turns': g.turns,
        'score_at_after': int(g.score),
        'game_over': g.game_over,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--anchor-pt', required=True,
                   help='sample_b_anchors.py output .pt')
    p.add_argument('--output', required=True)
    p.add_argument('--k', type=int, default=128,
                   help='Rollouts per (anchor, move).')
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--top-moves', type=int, default=6)
    p.add_argument('--fleet-size', type=int, default=512)
    p.add_argument('--max-anchors', type=int, default=0,
                   help='Process only the first N anchors. 0 = all.')
    p.add_argument('--device', default='mps')
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--forward-batch', type=int, default=128)
    p.add_argument('--legal-top-k', type=int, default=30,
                   help='Top-K legal moves to consider for top-N selection.')
    p.add_argument('--anchor-after-seed', type=int, default=9_000_000)
    p.add_argument('--continuation-seed-base', type=int, default=10_000_000)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # ── Load model ──
    print(f"\nLoading {args.model}...", flush=True)
    model, epoch = load_b_model(args.model, device,
                                  args.num_blocks, args.channels)
    print(f"  epoch={epoch}", flush=True)

    # ── Load anchors ──
    print(f"\nLoading {args.anchor_pt}...", flush=True)
    raw = torch.load(args.anchor_pt, weights_only=False)
    anchors_in = raw['anchors']
    if args.max_anchors > 0 and args.max_anchors < len(anchors_in):
        anchors_in = anchors_in[:args.max_anchors]
        print(f"  capping to first {args.max_anchors} anchors", flush=True)
    print(f"  {len(anchors_in)} anchors to mine", flush=True)

    # Assign IDs
    for i, a in enumerate(anchors_in):
        a['id'] = i

    # ── Phase A: compute top-N legal moves per anchor ──
    print(f"\nPhase A: B forward + top-{args.top_moves} legal moves...",
          flush=True)
    t0 = time.time()

    @torch.no_grad()
    def batched_forward(obs_np):
        ob_t = torch.from_numpy(obs_np).to(device)
        out = model(ob_t)
        if isinstance(out, tuple):
            out = out[0]
        return torch.softmax(out.float(), dim=-1).cpu().numpy()

    def _build_obs_for_anchor(a):
        return build_observation(
            np.asarray(a['board'], dtype=np.int8),
            np.array([nb[0][0] for nb in a['next_balls']], dtype=np.int64),
            np.array([nb[0][1] for nb in a['next_balls']], dtype=np.int64),
            np.array([nb[1] for nb in a['next_balls']], dtype=np.int64),
            int(a['num_next']),
        )

    anchor_topn = {}     # anchor_id -> [(move_int, prior), ...] sorted desc
    anchor_priors = {}   # anchor_id -> dict from _get_legal_priors_flat
    for start in range(0, len(anchors_in), args.forward_batch):
        sub = anchors_in[start:start + args.forward_batch]
        obs = np.stack([_build_obs_for_anchor(a) for a in sub])
        pols = batched_forward(obs)
        for k, a in enumerate(sub):
            priors = _get_legal_priors_flat(
                np.asarray(a['board'], dtype=np.int8),
                pols[k], args.legal_top_k)
            anchor_priors[a['id']] = priors
            if not priors or len(priors) < 2:
                anchor_topn[a['id']] = []
                continue
            sorted_moves = sorted(priors.items(), key=lambda kv: -kv[1])
            anchor_topn[a['id']] = sorted_moves[:args.top_moves]
    valid_ids = [a['id'] for a in anchors_in if len(anchor_topn[a['id']]) >= 2]
    print(f"  Phase A done in {time.time()-t0:.0f}s. "
          f"valid_anchors={len(valid_ids)}/{len(anchors_in)}", flush=True)

    # ── Phase B: realize afterstates per (anchor, move) ──
    print(f"\nPhase B: realize afterstates...", flush=True)
    t0 = time.time()
    afterstates = {}  # (anchor_id, move_int) -> afterstate or None
    for a in anchors_in:
        aid = a['id']
        if aid not in anchor_topn or not anchor_topn[aid]:
            continue
        anchor_seed = args.anchor_after_seed + aid * 7919
        for move, prior in anchor_topn[aid]:
            af = realize_afterstate(
                np.asarray(a['board'], dtype=np.int8),
                a['next_balls'],
                int(a.get('turn_origin', 0)),
                anchor_seed,
                int(move))
            if af is None:
                continue
            afterstates[(aid, int(move))] = {**af, 'prior': float(prior)}
    print(f"  Phase B done in {time.time()-t0:.0f}s. "
          f"afterstates={len(afterstates)}", flush=True)

    # ── Phase C: build task queue (K rollouts per (anchor, move)) ──
    print(f"\nPhase C: build task queue (K={args.k})...", flush=True)
    tasks = []
    for (aid, move), af in afterstates.items():
        cont_base = args.continuation_seed_base + aid * 7919 * 1000
        for k in range(args.k):
            tasks.append({
                'anchor_id': aid,
                'move': move,
                'k': k,
                'afterstate': af,
                'seed': cont_base + k,
            })
    # Pre-record dead-on-spawn outcomes
    per_move_outcomes = {}
    dead = [t for t in tasks if t['afterstate']['game_over']]
    for t in dead:
        per_move_outcomes.setdefault((t['anchor_id'], t['move']),
                                       []).append({
            'cap_hit': False,
            'score_gain': int(t['afterstate']['score_at_after']),
            'turns': 0,
            'k': t['k'],
        })
    tasks = [t for t in tasks if not t['afterstate']['game_over']]
    print(f"  total rollouts queued: {len(tasks):,} "
          f"(dead-on-spawn: {len(dead)})", flush=True)

    # ── Phase D: fleet rollouts ──
    print(f"\nPhase D: fleet rollouts (M={args.fleet_size}, "
          f"H={args.horizon})...", flush=True)
    M = args.fleet_size
    s_boards = np.zeros((M, 9, 9), dtype=np.int8)
    s_next_pos = np.zeros((M, 3, 2), dtype=np.int8)
    s_next_col = np.zeros((M, 3), dtype=np.int8)
    s_n_next = np.zeros(M, dtype=np.int8)
    s_scores = np.zeros(M, dtype=np.int32)
    s_turns = np.zeros(M, dtype=np.int32)
    s_game_overs = np.zeros(M, dtype=np.bool_)
    s_rng_states = np.zeros(M, dtype=np.uint64)
    s_target_end = np.zeros(M, dtype=np.int32)
    s_afterstate_turns = np.zeros(M, dtype=np.int32)
    s_afterstate_score = np.zeros(M, dtype=np.int32)
    s_task_idx = np.full(M, -1, dtype=np.int64)
    s_active = np.zeros(M, dtype=np.bool_)

    def load_slot(slot, task_idx):
        t = tasks[task_idx]
        af = t['afterstate']
        s_boards[slot] = af['board']
        for k in range(3):
            if k < len(af['next_balls']):
                pos, col = af['next_balls'][k]
                s_next_pos[slot, k, 0] = pos[0]
                s_next_pos[slot, k, 1] = pos[1]
                s_next_col[slot, k] = col
            else:
                s_next_pos[slot, k, 0] = 0
                s_next_pos[slot, k, 1] = 0
                s_next_col[slot, k] = 0
        s_n_next[slot] = min(len(af['next_balls']), 3)
        s_scores[slot] = 0
        s_turns[slot] = af['turns']
        s_game_overs[slot] = False
        s_rng_states[slot] = np.uint64(t['seed'])
        s_target_end[slot] = af['turns'] + args.horizon
        s_afterstate_turns[slot] = af['turns']
        s_afterstate_score[slot] = af['score_at_after']
        s_task_idx[slot] = task_idx
        s_active[slot] = True

    queue_idx = 0
    queue_len = len(tasks)
    for slot in range(M):
        if queue_idx < queue_len:
            load_slot(slot, queue_idx)
            queue_idx += 1

    t0 = time.time()
    n_steps = 0
    n_completed = 0
    last_log = t0

    while s_active.any():
        active_idx = np.where(s_active)[0]
        n_active = len(active_idx)

        active_boards = s_boards[active_idx]
        active_next_pos = s_next_pos[active_idx]
        active_next_col = s_next_col[active_idx]
        active_n_next = s_n_next[active_idx]
        obs_active = np.empty((n_active, 18, 9, 9), dtype=np.float32)
        build_obs_fleet_jit(active_boards, active_next_pos,
                             active_next_col, active_n_next, obs_active)

        pol_active = batched_forward(obs_active)

        active_scores = s_scores[active_idx].copy()
        active_turns = s_turns[active_idx].copy()
        active_game_overs = s_game_overs[active_idx].copy()
        active_completion = np.zeros(n_active, dtype=np.int8)
        active_rng = s_rng_states[active_idx].copy()

        step_fleet_jit(active_boards, active_next_pos, active_next_col,
                        active_n_next, active_scores, active_turns,
                        active_game_overs, active_completion,
                        pol_active.astype(np.float32), active_rng)

        s_boards[active_idx] = active_boards
        s_next_pos[active_idx] = active_next_pos
        s_next_col[active_idx] = active_next_col
        s_n_next[active_idx] = active_n_next
        s_scores[active_idx] = active_scores
        s_turns[active_idx] = active_turns
        s_game_overs[active_idx] = active_game_overs
        s_rng_states[active_idx] = active_rng

        for ai, slot in enumerate(active_idx):
            died = (active_completion[ai] == 1) or s_game_overs[slot]
            capped = (not died) and (s_turns[slot] >= s_target_end[slot])
            if died or capped:
                t = tasks[s_task_idx[slot]]
                outcome = {
                    'cap_hit': bool(capped),
                    'score_gain': int(s_afterstate_score[slot]
                                       + s_scores[slot]),
                    'turns': int(s_turns[slot]
                                  - s_afterstate_turns[slot]),
                    'k': t['k'],
                }
                per_move_outcomes.setdefault(
                    (t['anchor_id'], t['move']), []).append(outcome)
                n_completed += 1
                if queue_idx < queue_len:
                    load_slot(slot, queue_idx)
                    queue_idx += 1
                else:
                    s_active[slot] = False

        n_steps += 1
        now = time.time()
        if now - last_log > 15.0:
            elapsed = now - t0
            rate = n_completed / max(elapsed, 1e-3)
            remaining = (queue_len - queue_idx) + int(s_active.sum())
            eta = remaining / max(rate, 1e-3)
            print(f"  step {n_steps} | active={n_active} | "
                  f"completed={n_completed}/{queue_len} | "
                  f"rate={rate:.1f} r/s | "
                  f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s", flush=True)
            last_log = now

    print(f"  Phase D done in {time.time()-t0:.0f}s. "
          f"completed={n_completed} rollouts in {n_steps} steps", flush=True)

    # ── Phase E: aggregate per anchor with split-half ──
    print(f"\nPhase E: aggregate...", flush=True)
    out_anchors = []
    for a in anchors_in:
        aid = a['id']
        moves = anchor_topn.get(aid, [])
        if not moves:
            continue
        top_moves = []
        for rank_zero, (move, prior) in enumerate(moves):
            outcomes = per_move_outcomes.get((aid, int(move)), [])
            if not outcomes:
                continue
            # Split-half on k-index: even k -> half1, odd k -> half2
            h1 = [o for o in outcomes if o['k'] % 2 == 0]
            h2 = [o for o in outcomes if o['k'] % 2 == 1]
            def _cap(lst):
                return (float(sum(o['cap_hit'] for o in lst) / len(lst))
                        if lst else 0.0)
            top_moves.append({
                'rank': rank_zero + 1,
                'move_int': int(move),
                'prior': float(prior),
                'n_rollouts': len(outcomes),
                'cap_rate': _cap(outcomes),
                'cap_rate_half1': _cap(h1),
                'cap_rate_half2': _cap(h2),
                'mean_turns': float(np.mean(
                    [o['turns'] for o in outcomes])),
                'mean_score': float(np.mean(
                    [o['score_gain'] for o in outcomes])),
            })
        if len(top_moves) < 2:
            continue
        rec = {
            'anchor_idx': aid,
            'board': np.asarray(a['board'], dtype=np.int8),
            'next_balls': a['next_balls'],
            'top_moves': top_moves,
            'anchor_input_meta': {
                k: v for k, v in a.items()
                if k not in ('board', 'next_balls', 'id')
            },
        }
        out_anchors.append(rec)

    print(f"  aggregated {len(out_anchors)} anchors", flush=True)

    # Audit summary
    if out_anchors:
        cap_arr = np.array([
            max(m['cap_rate'] for m in r['top_moves'])
            for r in out_anchors])
        delta_arr = np.array([
            max(m['cap_rate'] for m in r['top_moves'])
            - sorted(r['top_moves'],
                      key=lambda m: m['rank'])[0]['cap_rate']
            for r in out_anchors])
        sh_agree = np.array([
            int(np.argmax([m['cap_rate_half1'] for m in r['top_moves']])
                == np.argmax([m['cap_rate_half2']
                                for m in r['top_moves']]))
            for r in out_anchors])
        print(f"\n  best_cap_rate: mean {cap_arr.mean():.3f}, "
              f"P50 {np.median(cap_arr):.3f}, "
              f"P90 {np.percentile(cap_arr, 90):.3f}", flush=True)
        print(f"  Δcap(best vs policy_top1): mean {delta_arr.mean():.3f}, "
              f"P50 {np.median(delta_arr):.3f}, "
              f"P90 {np.percentile(delta_arr, 90):.3f}", flush=True)
        print(f"  split-half winner agreement: "
              f"{sh_agree.mean()*100:.1f}%", flush=True)
        for thresh in (0.10, 0.15, 0.20, 0.25, 0.30):
            n_above = (delta_arr >= thresh).sum()
            n_above_stable = ((delta_arr >= thresh) & (sh_agree == 1)).sum()
            print(f"    Δcap≥{thresh}: {n_above} ({100*n_above/len(out_anchors):.1f}%) "
                  f"| stable: {n_above_stable} ({100*n_above_stable/len(out_anchors):.1f}%)",
                  flush=True)

    # Save
    out = {
        'args': vars(args),
        'source_anchor_args': raw.get('args', {}),
        'anchors': out_anchors,
        'summary': {
            'n_anchors_input': len(anchors_in),
            'n_anchors_kept': len(out_anchors),
            'n_rollouts': n_completed + len(dead),
        },
    }
    torch.save(out, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved {args.output} ({size_mb:.1f} MB)", flush=True)


if __name__ == '__main__':
    main()
