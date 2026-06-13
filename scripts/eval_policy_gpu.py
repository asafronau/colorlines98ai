"""GPU-resident batched policy eval — protocol v2 (docs/gpu_eval_engine_plan.md).

The whole game loop lives on the device: CC labels -> observation -> fp16 forward ->
masked argmax -> engine step (move/clear/spawn). The CPU touches only per-game init
(ColorLinesGame(seed).reset() — identical historical starting positions) and tiny
per-step alive/score reads. Spawn randomness = stateless per-(seed,turn) keys, so each
seed's game is deterministic INDEPENDENT of batch composition (golden test 4); scores
still depend on fp16 forward kernels, so pin (device, batch) per protocol, same as v1.

Ship gates passed (M5/mps): trace-injection bit-fidelity, argmax-legal equivalence,
spawn uniformity, batch-independence (scripts/test_gpu_engine_{golden,argmax_golden,rng}).

    PYTHONPATH=. python -m scripts.eval_policy_gpu --model M.pt \\
        --seed-start 775000 --seed-end 779999 --device mps --batch 512
"""
import os, sys, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from game.board import ColorLinesGame
from alphatrain.evaluate import load_model
from alphatrain.gpu_eval_engine import (
    GpuGames, choose_moves, seed_stream, NCELL,
    label_components_sv, build_observation_t,
)
from scripts.eval_policy import _stats


def _init_slot(st, slot, seed):
    g = ColorLinesGame(seed=seed)
    g.reset()
    st.boards[slot] = torch.from_numpy(g.board).to(st.dev)
    st.next_pos[slot] = 0
    st.next_col[slot] = 0
    for i, ((r, c), col) in enumerate(g.next_balls):
        st.next_pos[slot, i, 0] = r
        st.next_pos[slot, i, 1] = c
        st.next_col[slot, i] = col
    st.n_next[slot] = len(g.next_balls)
    st.score[slot] = 0
    st.turns[slot] = 0
    st.rng[slot] = seed_stream(seed)
    st.alive[slot] = True


@torch.no_grad()
def eval_policy_gpu(net, dev, dtype, seeds, batch=512, max_turns=1_000_000,
                    log_every=200):
    todo = list(seeds)
    results = {}
    B = min(batch, len(todo))
    st = GpuGames(B, dev)
    slot_seed = [None] * B
    nxt = 0
    for s in range(B):
        _init_slot(st, s, todo[nxt])
        slot_seed[s] = todo[nxt]
        nxt += 1

    fwd = evals = 0
    t0 = time.perf_counter()
    while any(s is not None for s in slot_seed):
        labels = label_components_sv(st.boards)
        obs = build_observation_t(st.boards, st.next_pos, st.next_col,
                                  st.n_next, labels=labels)
        out = net(obs.to(dtype))
        logits = (out[0] if isinstance(out, tuple) else out).float()
        moves, has = choose_moves(st.boards, logits, labels=labels)
        st.alive &= has                      # no legal move -> dead as-is
        mv = moves.clamp(min=0)
        st.step(mv // NCELL, mv % NCELL)
        st.alive &= st.turns < max_turns

        n_in_flight = sum(1 for s in slot_seed if s is not None)
        fwd += 1
        evals += n_in_flight
        alive_cpu = st.alive.cpu()
        if not bool(alive_cpu.all()):
            score_cpu = st.score.cpu()
            turns_cpu = st.turns.cpu()
            for s in range(B):
                if slot_seed[s] is not None and not bool(alive_cpu[s]):
                    results[slot_seed[s]] = (int(score_cpu[s]), int(turns_cpu[s]))
                    if nxt < len(todo):
                        _init_slot(st, s, todo[nxt])
                        slot_seed[s] = todo[nxt]
                        nxt += 1
                    else:
                        slot_seed[s] = None
        if fwd % log_every == 0:
            el = time.perf_counter() - t0
            print(f"  {len(results)}/{len(todo)} done, {fwd} steps "
                  f"(in-flight {n_in_flight}), {evals/el:.0f} evals/s, "
                  f"{el:.0f}s", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seed-start', type=int, required=True)
    p.add_argument('--seed-end', type=int, required=True, help='inclusive')
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--device', default='mps')
    p.add_argument('--max-turns', type=int, default=1_000_000)
    p.add_argument('--no-save-scores', action='store_true')
    a = p.parse_args()
    dev = torch.device(a.device)
    net, _ = load_model(a.model, dev, fp16=(dev.type != 'cpu'))
    net.train(False)
    dtype = next(net.parameters()).dtype
    seeds = list(range(a.seed_start, a.seed_end + 1))
    print(f"eval_policy_gpu (protocol v2): {len(seeds)} seeds "
          f"{a.seed_start}..{a.seed_end}, device={a.device} dtype={dtype} "
          f"batch={a.batch}", flush=True)
    t0 = time.perf_counter()
    res = eval_policy_gpu(net, dev, dtype, seeds, batch=a.batch,
                          max_turns=a.max_turns)
    dt = time.perf_counter() - t0
    print(f"\nDone: {len(res)} games in {dt:.0f}s", flush=True)
    _stats([s for s, _ in res.values()])
    if not a.no_save_scores:
        os.makedirs('logs/eval_scores', exist_ok=True)
        tag = os.path.splitext(os.path.basename(a.model))[0]
        out = (f'logs/eval_scores/{tag}_v2_{a.seed_start}_{a.seed_end}'
               f'_b{a.batch}_{a.device}.json')
        json.dump({str(k): [int(v[0]), int(v[1])] for k, v in res.items()},
                  open(out, 'w'))
        print(f"per-seed scores -> {out}", flush=True)


if __name__ == '__main__':
    main()
