"""Path B v2 — Phase 2: stratified anchor sampler from B_ep12 self-play.

Reads gen_b_selfplay.py output and emits ~3-5K anchor candidates stratified
across:
  - CRISIS: states from which the game died within --crisis-horizon turns.
            These are the floor-determining states we most need to learn.
  - DIVERSE: states from mid-game where the game survived for at least
             --diverse-survival turns afterward. Broad coverage of B's play.
  - (UNCERTAIN: high-entropy states. Requires B forward — added in a follow-up
    if needed; not in this v1 of the sampler.)

Output is a flat .pt file with one anchor per row, ready for downstream
K=128 mining:

{
  'args': {...},
  'anchors': [
    {
      'source_seed': int,
      'turn': int,
      'score': int,
      'board': int8 [9, 9],
      'next_pos': int8 [3, 2],
      'next_col': int8 [3],
      'n_next': int,
      'category': str,           # 'crisis' or 'diverse'
      'turns_to_death': int,     # -1 if game didn't die
      'final_score': int,
    },
    ...
  ]
}

Usage:
    python -m alphatrain.scripts.sample_b_anchors \\
        --input alphatrain/data/b_selfplay.pt \\
        --output alphatrain/data/b_anchors.pt \\
        --n-crisis 1500 --n-diverse 1500 \\
        --crisis-horizon 40 --diverse-survival 100 \\
        --seed 2026
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True,
                   help='gen_b_selfplay.py output .pt')
    p.add_argument('--output', required=True)
    # Bucket sizes (60/25/15 per ChatGPT 2026-05-18 review)
    p.add_argument('--n-crisis', type=int, default=3600)
    p.add_argument('--n-uncertain', type=int, default=1500)
    p.add_argument('--n-diverse', type=int, default=900)
    # Crisis filter
    p.add_argument('--crisis-horizon', type=int, default=40,
                   help='State is "crisis" if game died within this many '
                        'turns of the state.')
    # Diverse filters (replaces old --diverse-survival)
    p.add_argument('--diverse-score-progress', type=int, default=1000,
                   help='Diverse anchor must have final_score - '
                        'score_at_state >= this.')
    p.add_argument('--diverse-timeline-frac', type=float, default=0.3,
                   help='Diverse anchor must have '
                        '(final_turns - turn) / final_turns >= this. '
                        'For capped games, excludes the last (1 - this) '
                        'fraction of the trajectory.')
    # Uncertain filter (requires B forward)
    p.add_argument('--checkpoint', default=None,
                   help='B_ep12 checkpoint. Required if --n-uncertain > 0.')
    p.add_argument('--uncertain-top1-thresh', type=float, default=0.30,
                   help='State is "uncertain" if max legal-prob < this.')
    p.add_argument('--uncertain-batch-size', type=int, default=512)
    p.add_argument('--device', default='mps')
    # General
    p.add_argument('--min-turn', type=int, default=20,
                   help='Skip the first N turns of each game.')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    args = p.parse_args()

    if args.n_uncertain > 0 and args.checkpoint is None:
        raise SystemExit(
            "ERROR: --n-uncertain > 0 requires --checkpoint (B_ep12 "
            "needed for entropy computation). Set --n-uncertain 0 to skip.")

    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.input}...", flush=True)
    data = torch.load(args.input, weights_only=False)
    games = data['games']
    print(f"  {len(games)} games loaded", flush=True)

    # Three pools. Uncertain is filled in a second pass with B forward.
    crisis_pool = []
    diverse_pool = []
    uncertain_candidates = []  # all eligible states for B-entropy screen

    for g in games:
        seed = int(g['seed'])
        final_turns = int(g['final_turns'])
        died = bool(g['died'])
        capped = bool(g['capped'])
        final_score = int(g['final_score'])
        boards = g['boards']
        next_pos = g['next_pos']
        next_col = g['next_col']
        n_next = g['n_next']
        scores = g['scores']
        T = boards.shape[0]
        if T == 0:
            continue

        # Build per-state metadata. The trajectory's `turn` value isn't
        # explicitly stored per state — but we know each entry corresponds
        # to consecutive turns starting at 0 with --save-every spacing.
        # gen_b_selfplay saved every turn by default, so:
        #   turn_at_index[i] = i * save_every
        # but we may not know save_every from the file. Trust monotone
        # ascending turn for now: we can recover from the SOURCE turn by
        # using the saved (turn, score) tuples if available. The .pt
        # stored them via append, so position i corresponds to the i-th
        # recorded turn. The actual game turn requires multiplying by
        # save_every (default 1).
        save_every = data['args'].get('save_every', 1)

        for i in range(T):
            turn = i * save_every
            if turn < args.min_turn:
                continue
            # Skip the very last state (no decision to learn from)
            if i >= T - 1 and died:
                continue
            turns_to_death = (final_turns - turn) if died else -1

            # Build the anchor in the format phase1_oracle_fleet.py expects.
            nn_i = int(n_next[i])
            np_arr = next_pos[i]
            nc_arr = next_col[i]
            next_balls = [
                ((int(np_arr[k, 0]), int(np_arr[k, 1])), int(nc_arr[k]))
                for k in range(nn_i)
            ]
            rec = {
                'board': boards[i].copy().astype(np.int8),
                'next_balls': next_balls,
                'num_next': nn_i,
                'turn_origin': int(turn),
                'seed_origin': seed,
                'source_file': f'b_selfplay_seed{seed}',
                # extras for diagnostics:
                'score': int(scores[i]),
                'turns_to_death': int(turns_to_death),
                'final_score': final_score,
                'final_turns': final_turns,
                'capped_source_game': capped,
            }
            if died and turns_to_death <= args.crisis_horizon:
                rec['source_label'] = 'crisis'
                crisis_pool.append(rec)
                # Also eligible as uncertain candidate (any crisis state
                # may also be uncertain — pools overlap pre-stratification).
                uncertain_candidates.append(rec)
            else:
                # Diverse filter: F1 (score progress) AND F3 (timeline).
                # Adjust F3 for capped trajectories: exclude last
                # (1 - timeline_frac) of capped tails which may be circling.
                score_progress_ok = (
                    (final_score - rec['score']) >=
                    args.diverse_score_progress)
                if capped:
                    timeline_ok = (turn <
                                    args.diverse_timeline_frac * final_turns)
                else:
                    timeline_ok = (
                        (final_turns - turn) / max(final_turns, 1) >=
                        args.diverse_timeline_frac)
                if score_progress_ok and timeline_ok:
                    rec['source_label'] = 'diverse'
                    diverse_pool.append(rec)
                    uncertain_candidates.append(rec)

    print(f"\nPool sizes (full):", flush=True)
    print(f"  crisis pool: {len(crisis_pool)}", flush=True)
    print(f"  diverse pool: {len(diverse_pool)}", flush=True)

    # Sample crisis and diverse INDEPENDENTLY (no cannibalization).
    def _sample(pool, n):
        if len(pool) <= n:
            return list(pool)
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in idx]

    crisis = _sample(crisis_pool, args.n_crisis)
    diverse = _sample(diverse_pool, args.n_diverse)

    # ── Uncertain bucket: forward B on a SUBSAMPLE of union and pick
    # the top-N most uncertain by top1-legal-prob (rank-based, robust
    # to distribution shape). ──
    uncertain = []
    if args.n_uncertain > 0:
        import time
        from alphatrain.mcts import _get_legal_priors_flat
        from alphatrain.model import AlphaTrainNet
        from alphatrain.observation import build_observation

        device = torch.device(args.device)
        print(f"\nLoading {args.checkpoint} for uncertainty screen...",
              flush=True)
        ckpt = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
        state = ckpt['model']
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.replace('_orig_mod.', ''): v
                      for k, v in state.items()}
        model = AlphaTrainNet(num_blocks=args.num_blocks,
                              channels=args.channels).to(device)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                     if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        model.train(False)

        @torch.no_grad()
        def _forward(obs_np):
            ob = torch.from_numpy(obs_np).to(device)
            out = model(ob)
            if isinstance(out, tuple):
                out = out[0]
            return torch.softmax(out.float(), dim=-1).cpu().numpy()

        # Subsample to keep forward cost bounded. We don't need to
        # screen all 2M states — 50K random samples give us plenty of
        # variety to pick the N most uncertain.
        n_screen = max(args.n_uncertain * 30, 20_000)
        if n_screen >= len(uncertain_candidates):
            screen_set = uncertain_candidates
        else:
            idx = rng.choice(len(uncertain_candidates),
                              size=n_screen, replace=False)
            screen_set = [uncertain_candidates[i] for i in idx]
        print(f"  uncertainty screen: forwarding "
              f"{len(screen_set)} states (subsampled from "
              f"{len(uncertain_candidates)})...", flush=True)
        t0 = time.time()
        top1_probs = np.full(len(screen_set), 1.0, dtype=np.float32)
        for start in range(0, len(screen_set),
                            args.uncertain_batch_size):
            sub = screen_set[start:start + args.uncertain_batch_size]
            obs_batch = np.stack([
                build_observation(
                    rec['board'],
                    np.array([nb[0][0] for nb in rec['next_balls']],
                              dtype=np.int64),
                    np.array([nb[0][1] for nb in rec['next_balls']],
                              dtype=np.int64),
                    np.array([nb[1] for nb in rec['next_balls']],
                              dtype=np.int64),
                    rec['num_next'],
                )
                for rec in sub
            ])
            pol = _forward(obs_batch)
            for k, rec in enumerate(sub):
                priors = _get_legal_priors_flat(rec['board'], pol[k], 30)
                if priors:
                    total = sum(priors.values())
                    if total > 0:
                        top1_probs[start + k] = max(priors.values()) / total
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

        # Distribution diagnostics
        print(f"  top1_prob (renormalized legal): "
              f"mean={top1_probs.mean():.3f} "
              f"P10={np.percentile(top1_probs, 10):.3f} "
              f"P25={np.percentile(top1_probs, 25):.3f} "
              f"P50={np.percentile(top1_probs, 50):.3f} "
              f"P75={np.percentile(top1_probs, 75):.3f}",
              flush=True)

        # Pick the n_uncertain most uncertain (lowest top1_prob). This is
        # rank-based: robust to whatever distribution shape B's policy
        # has.
        order = np.argsort(top1_probs)
        picked = order[:args.n_uncertain]
        for k_idx in picked:
            rec = dict(screen_set[k_idx])  # shallow copy to override label
            rec['_top1_prob'] = float(top1_probs[k_idx])
            rec['source_label'] = 'uncertain'
            uncertain.append(rec)
        if uncertain:
            top1_uncertain = np.array([r['_top1_prob']
                                         for r in uncertain])
            print(f"  uncertain bucket top1_prob: "
                  f"max={top1_uncertain.max():.3f} "
                  f"(rank-based selection — most uncertain)",
                  flush=True)

    # De-duplicate by (seed, turn) — uncertain may overlap with
    # crisis/diverse since we sampled independently from the same union.
    seen = set()
    anchors = []
    for rec in crisis + uncertain + diverse:
        key = (rec.get('seed_origin'), rec.get('turn_origin'))
        if key in seen:
            continue
        seen.add(key)
        anchors.append(rec)
    rng.shuffle(anchors)

    print(f"\nSampled:", flush=True)
    print(f"  crisis:    {len(crisis)} (target {args.n_crisis})",
          flush=True)
    print(f"  uncertain: {len(uncertain)} (target {args.n_uncertain})",
          flush=True)
    print(f"  diverse:   {len(diverse)} (target {args.n_diverse})",
          flush=True)
    print(f"  total anchors: {len(anchors)}", flush=True)

    # Score / turn distribution of the source games
    source_seeds = list({a['seed_origin'] for a in anchors})
    print(f"  unique source seeds: {len(source_seeds)}", flush=True)

    final_scores = np.array([a['final_score'] for a in anchors])
    final_turns = np.array([a['final_turns'] for a in anchors])
    print(f"  source game final_score: "
          f"mean {final_scores.mean():.0f} "
          f"median {np.median(final_scores):.0f}", flush=True)
    print(f"  source game final_turns: "
          f"mean {final_turns.mean():.0f} "
          f"median {np.median(final_turns):.0f}", flush=True)

    out = {
        'args': vars(args),
        'source_input_args': data.get('args', {}),
        'anchors': anchors,
    }
    torch.save(out, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved {args.output} ({size_mb:.1f} MB)", flush=True)


if __name__ == '__main__':
    main()
