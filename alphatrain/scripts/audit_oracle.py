"""Oracle label audit: K=128 rollouts on high-margin anchors using B_ep12.

For each anchor with delta_cap >= threshold (default 0.15):
  1. Compute B_ep12's top-1 LEGAL move from anchor state.
  2. Identify oracle's labeled best move (max cap_rate among top-6).
  3. If B agrees with oracle: count as agreement (no rollouts).
  4. If B disagrees: run K=128 fresh rollouts from BOTH moves' afterstates,
     using B_ep12 as the rollout policy.
  5. Compare cap_rates and mean_turns.

Outputs:
  - {output_stem}.pt   structured per-anchor results
  - {output_stem}.md   human-readable summary

Usage:
    python -m alphatrain.scripts.audit_oracle \\
        --checkpoint alphatrain/data/b_smoke_epoch_12.pt \\
        --oracle-tensor alphatrain/data/phase1_oracle_path_b.pt \\
        --output alphatrain/data/audit_oracle_results \\
        --margin-threshold 0.15 --k-rollouts 128 --horizon 300 \\
        --fleet-size 512 --device mps
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
    epoch = ckpt.get('epoch', '?')
    return model, epoch


def realize_afterstate(board, next_balls, turn_origin, seed, move):
    """Apply `move` (flat src*81+tgt) to the anchor; return afterstate dict or None."""
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
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--oracle-tensor', required=True)
    p.add_argument('--output', required=True,
                   help='Stem for .pt and .md outputs (no extension)')
    p.add_argument('--margin-threshold', type=float, default=0.15)
    p.add_argument('--k-rollouts', type=int, default=128)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--fleet-size', type=int, default=512)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--device', default='mps')
    p.add_argument('--anchor-after-seed', type=int, default=9_000_000)
    p.add_argument('--continuation-seed-base', type=int, default=10_000_000)
    p.add_argument('--forward-batch', type=int, default=128,
                   help='Batch size for B forward passes (used for picking '
                        "B's top-1 legal move and for rollout policy probs).")
    p.add_argument('--legal-top-k', type=int, default=30,
                   help='Top-K legal moves to consider for B argmax.')
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # ── Load model ──
    print(f"\nLoading {args.checkpoint}...", flush=True)
    model, epoch = load_b_model(args.checkpoint, device,
                                 args.num_blocks, args.channels)
    print(f"  epoch={epoch}", flush=True)

    # ── Load oracle tensor ──
    print(f"\nLoading {args.oracle_tensor}...", flush=True)
    raw = torch.load(args.oracle_tensor, weights_only=False)
    boards = raw['boards'].numpy()
    next_pos = raw['next_pos'].numpy()
    next_col = raw['next_col'].numpy()
    n_next = raw['n_next'].numpy()
    actions = raw['actions'].numpy()
    cap_rates = raw['cap_rates'].numpy()
    delta_cap = raw['delta_cap'].numpy()
    turn_origin = raw['turn_origin'].numpy()

    keep = delta_cap >= args.margin_threshold
    idx = np.where(keep)[0]
    print(f"  high-margin anchors (Δcap >= {args.margin_threshold}): "
          f"{len(idx)} / {len(boards)}", flush=True)

    # ── Phase A: compute B's top-1 legal pick per anchor ──
    print(f"\nPhase A: compute B's top-1 legal pick per anchor "
          f"(forward batch={args.forward_batch})...", flush=True)
    t0 = time.time()

    def _obs_for_anchor(i):
        nn_i = int(n_next[i])
        return build_observation(
            boards[i],
            next_pos[i, :nn_i, 0].astype(np.int64),
            next_pos[i, :nn_i, 1].astype(np.int64),
            next_col[i, :nn_i].astype(np.int64),
            nn_i,
        )

    @torch.no_grad()
    def batched_forward(obs_np):
        ob_t = torch.from_numpy(obs_np).to(device)
        out = model(ob_t)
        if isinstance(out, tuple):
            out = out[0]
        return torch.softmax(out.float(), dim=-1).cpu().numpy()

    anchor_specs = []
    for start in range(0, len(idx), args.forward_batch):
        sub = idx[start:start + args.forward_batch]
        obs = np.stack([_obs_for_anchor(i) for i in sub])
        pols = batched_forward(obs)
        for k, i in enumerate(sub):
            cap_row = cap_rates[i]
            oracle_slot = int(np.argmax(cap_row))
            oracle_move = int(actions[i, oracle_slot])

            priors = _get_legal_priors_flat(
                boards[i].astype(np.int8), pols[k], args.legal_top_k)
            if not priors:
                continue
            b_move, b_prior = max(priors.items(), key=lambda kv: kv[1])

            anchor_specs.append({
                'anchor_idx': int(i),
                'oracle_move': oracle_move,
                'oracle_cap_orig_k32': float(cap_row[oracle_slot]),
                'b_move': int(b_move),
                'b_prior': float(b_prior),
                'agree': int(b_move) == oracle_move,
                'b_in_top6': int(b_move) in set(int(a) for a in actions[i]
                                                  if a >= 0),
                'delta_cap': float(delta_cap[i]),
                'turn_origin': int(turn_origin[i]),
            })
        if (start + args.forward_batch) % (args.forward_batch * 8) == 0:
            print(f"  [{min(start+args.forward_batch, len(idx))}/{len(idx)}] "
                  f"in {time.time()-t0:.0f}s", flush=True)
    print(f"  Phase A done in {time.time()-t0:.0f}s. "
          f"Specs collected: {len(anchor_specs)}", flush=True)

    # ── Agreement summary ──
    n_agree = sum(1 for s in anchor_specs if s['agree'])
    n_b_in_top6 = sum(1 for s in anchor_specs if s['b_in_top6'])
    print(f"\nB ↔ oracle agreement (high-margin): "
          f"{n_agree}/{len(anchor_specs)} "
          f"= {100*n_agree/max(1, len(anchor_specs)):.1f}%", flush=True)
    print(f"  B's top-1 lies inside original top-6: "
          f"{n_b_in_top6}/{len(anchor_specs)} "
          f"= {100*n_b_in_top6/max(1, len(anchor_specs)):.1f}%", flush=True)

    # ── Phase B: realize afterstates for both moves per DISAGREEING anchor ──
    print(f"\nPhase B: realize afterstates for disagreeing anchors...",
          flush=True)
    disagreement = [s for s in anchor_specs if not s['agree']]
    print(f"  disagreeing anchors: {len(disagreement)}", flush=True)
    t0 = time.time()
    afterstates = {}  # (anchor_idx, tag) -> afterstate
    for s in disagreement:
        i = s['anchor_idx']
        seed = args.anchor_after_seed + i * 7919
        nbs = [(tuple(next_pos[i, k]), int(next_col[i, k]))
               for k in range(int(n_next[i]))]
        for tag, move in (('oracle', s['oracle_move']),
                            ('B', s['b_move'])):
            af = realize_afterstate(boards[i], nbs, s['turn_origin'],
                                      seed, move)
            if af is None:
                continue
            afterstates[(i, tag)] = af
    print(f"  Phase B done in {time.time()-t0:.0f}s. "
          f"Afterstates: {len(afterstates)}", flush=True)

    # ── Phase C: build rollout task queue ──
    print(f"\nPhase C: build task queue (K={args.k_rollouts})...",
          flush=True)
    tasks = []
    for (i, tag), af in afterstates.items():
        cont_base = args.continuation_seed_base + i * 7919 * 1000
        # Different seed range per tag so the two arms don't share seeds.
        offset = 0 if tag == 'oracle' else args.k_rollouts
        for k in range(args.k_rollouts):
            tasks.append({
                'anchor_idx': i,
                'tag': tag,
                'afterstate': af,
                'seed': cont_base + offset + k,
            })

    # Pre-record dead-on-spawn outcomes
    per_move_outcomes = {}
    dead = [t for t in tasks if t['afterstate']['game_over']]
    for t in dead:
        per_move_outcomes.setdefault((t['anchor_idx'], t['tag']),
                                       []).append({
            'cap_hit': False,
            'score_gain': int(t['afterstate']['score_at_after']),
            'turns': 0,
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
                        pol_active, active_rng)

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
                }
                per_move_outcomes.setdefault(
                    (t['anchor_idx'], t['tag']), []).append(outcome)
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

    # ── Phase E: aggregate per anchor ──
    print(f"\nPhase E: aggregate...", flush=True)
    results = []
    n_b_win = 0
    n_oracle_win = 0
    n_tie = 0
    cap_diff = []
    turn_diff = []
    for s in disagreement:
        i = s['anchor_idx']
        o_out = per_move_outcomes.get((i, 'oracle'), [])
        b_out = per_move_outcomes.get((i, 'B'), [])
        if not o_out or not b_out:
            continue
        o_cap = float(sum(x['cap_hit'] for x in o_out) / len(o_out))
        b_cap = float(sum(x['cap_hit'] for x in b_out) / len(b_out))
        o_turns = float(np.mean([x['turns'] for x in o_out]))
        b_turns = float(np.mean([x['turns'] for x in b_out]))
        if b_cap > o_cap:
            n_b_win += 1
        elif b_cap < o_cap:
            n_oracle_win += 1
        else:
            n_tie += 1
        cap_diff.append(b_cap - o_cap)
        turn_diff.append(b_turns - o_turns)
        results.append({
            'anchor_idx': i,
            'delta_cap_orig': s['delta_cap'],
            'oracle_move': s['oracle_move'],
            'b_move': s['b_move'],
            'b_in_top6': s['b_in_top6'],
            'b_prior': s['b_prior'],
            'oracle_cap_orig_k32': s['oracle_cap_orig_k32'],
            'oracle_cap_k128': o_cap,
            'b_cap_k128': b_cap,
            'oracle_turns_k128': o_turns,
            'b_turns_k128': b_turns,
            'n_rollouts_oracle': len(o_out),
            'n_rollouts_b': len(b_out),
            'cap_diff': b_cap - o_cap,
            'turn_diff': b_turns - o_turns,
            'winner': ('B' if b_cap > o_cap
                        else ('oracle' if b_cap < o_cap else 'tie')),
        })

    n_evaluated = n_b_win + n_oracle_win + n_tie
    cap_diff_arr = np.array(cap_diff) if cap_diff else np.zeros(1)
    turn_diff_arr = np.array(turn_diff) if turn_diff else np.zeros(1)

    summary = {
        'checkpoint': args.checkpoint,
        'oracle_tensor': args.oracle_tensor,
        'margin_threshold': args.margin_threshold,
        'k_rollouts': args.k_rollouts,
        'horizon': args.horizon,
        'n_high_margin_anchors': len(idx),
        'n_specs_collected': len(anchor_specs),
        'n_agreement': n_agree,
        'agreement_rate': n_agree / max(1, len(anchor_specs)),
        'n_disagreement_with_outcomes': n_evaluated,
        'b_wins': n_b_win,
        'oracle_wins': n_oracle_win,
        'ties': n_tie,
        'b_win_rate_on_disagreements': n_b_win / max(1, n_evaluated),
        'oracle_win_rate_on_disagreements': n_oracle_win / max(1, n_evaluated),
        'cap_diff_mean': float(cap_diff_arr.mean()),
        'cap_diff_median': float(np.median(cap_diff_arr)),
        'turn_diff_mean': float(turn_diff_arr.mean()),
        'turn_diff_median': float(np.median(turn_diff_arr)),
    }

    out_pt = args.output + '.pt'
    torch.save({'summary': summary, 'results': results,
                 'anchor_specs': anchor_specs}, out_pt)
    print(f"\nSaved {out_pt}", flush=True)

    # Markdown
    out_md = args.output + '.md'
    lines = []
    lines.append('# Oracle audit results')
    lines.append('')
    lines.append(f"Checkpoint: `{os.path.basename(args.checkpoint)}`  ")
    lines.append(f"Oracle tensor: `{os.path.basename(args.oracle_tensor)}`  ")
    lines.append(f"K rollouts per move: **{args.k_rollouts}**  ")
    lines.append(f"Horizon per rollout: **{args.horizon}**  ")
    lines.append(f"Margin threshold: **Δcap >= {args.margin_threshold}**  ")
    lines.append('')
    lines.append('## Agreement')
    lines.append(f"- High-margin anchors: **{len(idx)}**")
    lines.append(f"- Specs collected: **{len(anchor_specs)}**")
    lines.append(f"- B-vs-oracle agreement: **{n_agree} / {len(anchor_specs)} "
                 f"= {100*n_agree/max(1, len(anchor_specs)):.1f}%**")
    lines.append(f"- B's top-1 inside original top-6: "
                 f"{n_b_in_top6}/{len(anchor_specs)} "
                 f"= {100*n_b_in_top6/max(1, len(anchor_specs)):.1f}%")
    lines.append('')
    lines.append(f"## Deep K={args.k_rollouts} judgment on disagreements")
    lines.append(f"- B wins (higher cap_rate): "
                 f"**{n_b_win} / {n_evaluated} "
                 f"= {100*n_b_win/max(1, n_evaluated):.1f}%**")
    lines.append(f"- Oracle wins: **{n_oracle_win} / {n_evaluated} "
                 f"= {100*n_oracle_win/max(1, n_evaluated):.1f}%**")
    lines.append(f"- Ties: {n_tie}")
    lines.append(f"- Mean Δcap_rate (B − oracle): "
                 f"{summary['cap_diff_mean']:+.4f}")
    lines.append(f"- Median Δcap_rate: {summary['cap_diff_median']:+.4f}")
    lines.append(f"- Mean Δturns (B − oracle): "
                 f"{summary['turn_diff_mean']:+.1f}")
    lines.append('')
    lines.append('### Interpretation')
    if n_b_win > 1.2 * n_oracle_win:
        lines.append("- **B wins materially more often than oracle.** "
                     "Original K=32 oracle labels appear systematically wrong "
                     "on hard anchors → re-mine with deeper K or B-derived "
                     "anchors before training again.")
    elif n_oracle_win > 1.2 * n_b_win:
        lines.append("- **Oracle still wins under deeper judgment.** "
                     "Labels are correct; the C/D training objective failed "
                     "to use the signal usefully.")
    else:
        lines.append("- **B and oracle are roughly tied under deeper "
                     "judgment.** Labels are noisy at this margin tier; "
                     "neither side has a clear edge.")
    lines.append('')
    lines.append('## Stats')
    lines.append(f"```\n{summary}\n```")

    with open(out_md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved {out_md}", flush=True)

    # Console summary
    print(f"\n{'='*64}\nAUDIT SUMMARY\n{'='*64}", flush=True)
    print(f"Agreement: {n_agree}/{len(anchor_specs)} "
          f"({100*n_agree/max(1, len(anchor_specs)):.1f}%)", flush=True)
    print(f"Deep K={args.k_rollouts} on disagreements ({n_evaluated}):",
          flush=True)
    print(f"  B wins:      {n_b_win} "
          f"({100*n_b_win/max(1, n_evaluated):.1f}%)", flush=True)
    print(f"  Oracle wins: {n_oracle_win} "
          f"({100*n_oracle_win/max(1, n_evaluated):.1f}%)", flush=True)
    print(f"  Ties:        {n_tie}", flush=True)
    print(f"  Mean Δcap (B - oracle): {summary['cap_diff_mean']:+.4f}",
          flush=True)


if __name__ == '__main__':
    main()
