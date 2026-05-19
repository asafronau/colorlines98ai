"""Build the compact Path B training tensor from a combined oracle .pt.

Input:  alphatrain/data/phase1_oracle_combined.pt (16,897 anchors, per_move dict)
Output: alphatrain/data/phase1_oracle_path_b.pt (flat tensors)

Schema (all CPU tensors; loader uploads to GPU once at start):

  boards:        int8   [N, 9, 9]
  next_pos:      int8   [N, 3, 2]
  next_col:      int8   [N, 3]
  n_next:        int8   [N]
  actions:       int64  [N, 6]     (top-6 flat src*81+tgt; pad -1)
  cap_rates:     float32[N, 6]     (cap rate per move; pad 0.0)
  mean_turns:    float32[N, 6]     (audit / sanity; pad 0.0)
  n_moves:       int8   [N]        (2..6 actual move count)
  delta_cap:     float32[N]        (max cap_rate − rank-1 cap_rate; ≥0)
  turn_origin:   int32  [N]
  is_crisis:     bool   [N]        (source_label == 'crisis')

Validation per move (hard, errors out if any anchor fails):
  - 0 <= action < 6561
  - board[sr, sc] > 0   (source has a ball)
  - board[tr, tc] == 0  (target empty)
Plus a SAMPLE reachability audit (BFS via game engine) on N_AUDIT random anchors.
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
import torch

from game.board import ColorLinesGame


def decode_action(a: int) -> tuple[int, int, int, int]:
    """Returns (sr, sc, tr, tc) from flat src*81 + tgt encoding."""
    sr = a // 81 // 9
    sc = a // 81 % 9
    tr = a % 81 // 9
    tc = a % 81 % 9
    return sr, sc, tr, tc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True,
                   help='Path to combined oracle .pt (output of '
                        'combine_oracle_datasets.py)')
    p.add_argument('--output', required=True)
    p.add_argument('--top-k', type=int, default=6,
                   help='Slots per anchor (will pad shorter anchors)')
    p.add_argument('--n-audit', type=int, default=500,
                   help='Number of anchors for BFS reachability audit')
    p.add_argument('--audit-seed', type=int, default=2026)
    args = p.parse_args()

    print(f"Loading {args.input}...", flush=True)
    raw = torch.load(args.input, weights_only=False)
    results = raw['results']
    print(f"  {len(results)} anchors loaded", flush=True)

    K = args.top_k
    N = len(results)

    boards = np.zeros((N, 9, 9), dtype=np.int8)
    next_pos = np.zeros((N, 3, 2), dtype=np.int8)
    next_col = np.zeros((N, 3), dtype=np.int8)
    n_next = np.zeros(N, dtype=np.int8)
    actions = np.full((N, K), -1, dtype=np.int64)
    cap_rates = np.zeros((N, K), dtype=np.float32)
    mean_turns = np.zeros((N, K), dtype=np.float32)
    n_moves = np.zeros(N, dtype=np.int8)
    delta_cap = np.zeros(N, dtype=np.float32)
    turn_origin = np.zeros(N, dtype=np.int32)
    is_crisis = np.zeros(N, dtype=bool)

    t0 = time.time()
    n_invalid = 0
    invalid_reasons = {'range': 0, 'src_empty': 0, 'tgt_occupied': 0,
                        'short': 0}

    for i, r in enumerate(results):
        b = np.asarray(r['anchor_board'], dtype=np.int8)
        boards[i] = b
        nbs = r['anchor_next_balls']
        nn = min(int(r['anchor_n_next']), 3)
        n_next[i] = nn
        for k in range(nn):
            pos, col = nbs[k]
            next_pos[i, k, 0] = pos[0]
            next_pos[i, k, 1] = pos[1]
            next_col[i, k] = col
        turn_origin[i] = int(r.get('turn_origin', 0))
        is_crisis[i] = (r.get('source_label', 'unknown') == 'crisis')

        pm = r['per_move']
        # Sort by rank ASC so slot 0 = policy's top-1
        sorted_pm = sorted(pm.items(), key=lambda kv: kv[1]['rank'])
        if len(sorted_pm) < 2:
            invalid_reasons['short'] += 1
            n_invalid += 1
            continue

        m_count = 0
        rank1_cap = None
        best_cap = -1.0
        for act_int, mv in sorted_pm[:K]:
            a = int(act_int)
            if not (0 <= a < 6561):
                invalid_reasons['range'] += 1
                n_invalid += 1
                raise ValueError(
                    f"Anchor {i} action {a} out of range [0, 6561)")
            sr, sc, tr, tc = decode_action(a)
            if b[sr, sc] == 0:
                invalid_reasons['src_empty'] += 1
                n_invalid += 1
                raise ValueError(
                    f"Anchor {i} action {a}: source ({sr},{sc}) is empty "
                    f"(board value {b[sr, sc]})")
            if b[tr, tc] != 0:
                invalid_reasons['tgt_occupied'] += 1
                n_invalid += 1
                raise ValueError(
                    f"Anchor {i} action {a}: target ({tr},{tc}) occupied "
                    f"(board value {b[tr, tc]})")
            actions[i, m_count] = a
            cap_rates[i, m_count] = float(mv['cap_rate'])
            mean_turns[i, m_count] = float(mv['mean_turns'])
            if mv['rank'] == 1:
                rank1_cap = float(mv['cap_rate'])
            if mv['cap_rate'] > best_cap:
                best_cap = float(mv['cap_rate'])
            m_count += 1
        n_moves[i] = m_count
        if rank1_cap is None:
            # No rank=1 in top-K (very unlikely); use slot 0 as fallback
            rank1_cap = cap_rates[i, 0]
        delta_cap[i] = max(0.0, best_cap - rank1_cap)

        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{N}] in {time.time()-t0:.0f}s", flush=True)

    print(f"  built tensors in {time.time()-t0:.0f}s. "
          f"shorts skipped: {invalid_reasons['short']}", flush=True)

    # Drop short anchors (n_moves < 2)
    keep = n_moves >= 2
    n_kept = int(keep.sum())
    if n_kept < N:
        print(f"  dropping {N - n_kept} anchors with <2 moves", flush=True)
        boards = boards[keep]
        next_pos = next_pos[keep]
        next_col = next_col[keep]
        n_next = n_next[keep]
        actions = actions[keep]
        cap_rates = cap_rates[keep]
        mean_turns = mean_turns[keep]
        n_moves = n_moves[keep]
        delta_cap = delta_cap[keep]
        turn_origin = turn_origin[keep]
        is_crisis = is_crisis[keep]
    N = n_kept

    # ── BFS reachability audit on a random sample ──
    print(f"\nReachability audit on {args.n_audit} random anchors...",
          flush=True)
    rng = np.random.default_rng(args.audit_seed)
    sample = rng.choice(N, size=min(args.n_audit, N), replace=False)
    audit_fail = 0
    t0 = time.time()
    for s in sample:
        b = boards[s]
        nbs = [(tuple(next_pos[s, k]), int(next_col[s, k]))
               for k in range(int(n_next[s]))]
        for k in range(int(n_moves[s])):
            a = int(actions[s, k])
            sr, sc, tr, tc = decode_action(a)
            g = ColorLinesGame(seed=0)
            g.reset(board=b.copy(), next_balls=list(nbs))
            # Validate that source-to-target is reachable via BFS
            mv_result = g.move((sr, sc), (tr, tc))
            if not mv_result.get('valid', False):
                audit_fail += 1
                print(f"  AUDIT FAIL: anchor #{s}, action={a} "
                      f"({sr},{sc})->({tr},{tc}), reason="
                      f"{mv_result.get('reason', 'unknown')}", flush=True)
    print(f"  audited {len(sample)} anchors × ~{n_moves[sample].mean():.1f} "
          f"moves in {time.time()-t0:.0f}s. failures: {audit_fail}",
          flush=True)
    if audit_fail > 0:
        raise RuntimeError(f"{audit_fail} reachability failures — STOP. "
                            f"Action indices may be misaligned with boards.")

    # ── Distribution sanity ──
    print(f"\n=== Tensor stats ===", flush=True)
    print(f"  N anchors: {N}", flush=True)
    print(f"  n_moves: mean {n_moves.mean():.2f}, "
          f"min {n_moves.min()}, max {n_moves.max()}", flush=True)
    print(f"  n_moves==6: {(n_moves == 6).sum()} "
          f"({100*(n_moves == 6).sum()/N:.1f}%)", flush=True)
    print(f"  delta_cap: mean {delta_cap.mean():.4f}, "
          f"P50 {np.percentile(delta_cap, 50):.4f}, "
          f"P90 {np.percentile(delta_cap, 90):.4f}", flush=True)
    print(f"  >=0.05: {(delta_cap >= 0.05).sum()} "
          f"({100*(delta_cap >= 0.05).sum()/N:.1f}%)", flush=True)
    print(f"  >=0.15: {(delta_cap >= 0.15).sum()} "
          f"({100*(delta_cap >= 0.15).sum()/N:.1f}%)", flush=True)
    print(f"  is_crisis: {is_crisis.sum()} "
          f"({100*is_crisis.sum()/N:.1f}%)", flush=True)

    # ── Save ──
    out = {
        'boards': torch.from_numpy(boards),
        'next_pos': torch.from_numpy(next_pos),
        'next_col': torch.from_numpy(next_col),
        'n_next': torch.from_numpy(n_next),
        'actions': torch.from_numpy(actions),
        'cap_rates': torch.from_numpy(cap_rates),
        'mean_turns': torch.from_numpy(mean_turns),
        'n_moves': torch.from_numpy(n_moves),
        'delta_cap': torch.from_numpy(delta_cap),
        'turn_origin': torch.from_numpy(turn_origin),
        'is_crisis': torch.from_numpy(is_crisis),
        'meta': {
            'source': args.input,
            'top_k': K,
            'n_anchors': N,
            'audit_n': len(sample),
            'audit_failures': 0,
        },
    }
    torch.save(out, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved {args.output} ({size_mb:.1f} MB)", flush=True)


if __name__ == '__main__':
    main()
