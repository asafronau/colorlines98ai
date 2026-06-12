"""Fast single-process batched policy-only eval — no workers, no inference server.

Policy play is just (build obs -> forward -> argmax legal -> move) per game, which
batches directly: hold B games in flight, do ONE batch-B forward per step, refill a
slot when a game dies. This gives a clean, large, CONSTANT GPU batch with zero IPC,
vs eval_parallel's N-process server whose batch is the accidental count of workers
mid-request. Same greedy-argmax-legal policy as eval_parallel --policy-only.

fp16 is batch-dependent (the score depends on B via rounding), but reproducible for a
fixed (seeds, batch). Use --fp32 for batch-invariant / config-independent scores.

    PYTHONPATH=. python scripts/eval_policy.py --model M.pt --seed-start 50000 --seed-end 50300 --batch 256
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.evaluate import load_model
from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
from scripts.batched_rollout import _decode


class _Slot:
    __slots__ = ('seed', 'game')

    def __init__(self, seed):
        self.seed = seed
        self.game = ColorLinesGame(seed=seed)
        self.game.reset()


@torch.no_grad()
def eval_policy(net, dev, dtype, seeds, batch=256, max_turns=1_000_000, log_every=10000):
    todo, nxt, results = list(seeds), 0, {}

    def make_slot():
        nonlocal nxt
        if nxt < len(todo):
            s = _Slot(todo[nxt]); nxt += 1; return s
        return None

    slots = [s for s in (make_slot() for _ in range(batch)) if s is not None]
    fwd = evals = 0
    t0 = time.perf_counter()
    while slots:
        obs = np.stack([_build_obs_for_game(s.game) for s in slots])
        logits = net(torch.from_numpy(obs).to(dev, dtype)).float().cpu().numpy()
        fwd += 1; evals += len(slots)
        survivors, finalized = [], 0
        for i, s in enumerate(slots):
            priors = _get_legal_priors_flat(s.game.board, logits[i], 30)
            dead = not priors
            if not dead:
                mv = _decode(max(priors.items(), key=lambda x: x[1])[0])
                dead = (not s.game.move(*mv)['valid']) or s.game.game_over \
                    or s.game.turns >= max_turns
            if dead:
                results[s.seed] = (int(s.game.score), int(s.game.turns))
                finalized += 1
            else:
                survivors.append(s)
        for _ in range(finalized):
            ns = make_slot()
            if ns is not None:
                survivors.append(ns)
        slots = survivors
        if fwd % log_every == 0:
            el = time.perf_counter() - t0
            print(f"  {len(results)}/{len(todo)} done, {fwd} fwd "
                  f"(bs~{batch}), {evals/el:.0f} evals/s, {el:.0f}s", flush=True)
    return results


def _stats(scores):
    a = np.array(sorted(scores))
    n = len(a)
    p = {q: a[min(n - 1, int(q / 100 * n))] for q in (1, 5, 10, 25, 50, 75, 90, 95)}
    print(f"\n  MEAN |  {a.mean():.0f}\n")
    print(f"  Pol stats ({n} games):")
    print(f"    min={a.min()}  max={a.max()}  mean={a.mean():.0f}")
    print(f"    P1={p[1]}  P5={p[5]}  P10={p[10]}  P25={p[25]}  P50={p[50]}  "
          f"P75={p[75]}  P90={p[90]}  P95={p[95]}")
    print(f"    <500: {(a<500).sum()} ({100*(a<500).mean():.1f}%)  "
          f"<1000: {(a<1000).sum()} ({100*(a<1000).mean():.1f}%)  "
          f">5000: {(a>5000).sum()} ({100*(a>5000).mean():.0f}%)  "
          f">10000: {(a>10000).sum()} ({100*(a>10000).mean():.0f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seed-start', type=int, required=True)
    p.add_argument('--seed-end', type=int, required=True, help='inclusive')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--device', default='mps')
    p.add_argument('--fp32', action='store_true',
                   help='Batch-invariant (config-independent) scores; slower.')
    p.add_argument('--max-turns', type=int, default=1_000_000)
    p.add_argument('--no-save-scores', action='store_true',
                   help='Skip the per-seed score JSON (saved by default to '
                        'logs/eval_scores/<model>_<start>_<end>.json — enables '
                        'PAIRED model comparisons on a common seed list, which '
                        'cancel per-seed luck and are far more sensitive than '
                        'comparing aggregate medians).')
    a = p.parse_args()
    dev = torch.device(a.device)
    fp16 = (not a.fp32) and dev.type != 'cpu'
    net, _ = load_model(a.model, dev, fp16=fp16)
    dtype = next(net.parameters()).dtype
    seeds = list(range(a.seed_start, a.seed_end + 1))
    print(f"eval_policy: {len(seeds)} seeds {a.seed_start}..{a.seed_end}, "
          f"device={a.device} dtype={dtype} batch={a.batch}", flush=True)
    t0 = time.perf_counter()
    res = eval_policy(net, dev, dtype, seeds, batch=a.batch, max_turns=a.max_turns)
    dt = time.perf_counter() - t0
    print(f"\nDone: {len(res)} games in {dt:.0f}s", flush=True)
    _stats([s for s, _ in res.values()])
    if not a.no_save_scores:
        import json
        os.makedirs('logs/eval_scores', exist_ok=True)
        tag = os.path.splitext(os.path.basename(a.model))[0]
        out = f'logs/eval_scores/{tag}_{a.seed_start}_{a.seed_end}.json'
        json.dump({str(k): [int(v[0]), int(v[1])] for k, v in res.items()},
                  open(out, 'w'))
        print(f"per-seed scores -> {out}", flush=True)


if __name__ == '__main__':
    main()
