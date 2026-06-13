"""Steady-state throughput of the GPU eval engine across batch sizes.

Fills a full batch from real pillar-strength game seeds, then times N steps of the
real loop (label -> obs -> fp16 forward -> argmax -> engine step). pillar3f games
rarely die before ~10k turns, so in-flight stays == batch for the whole window =
a clean steady-state reading (no drain contamination). Reports evals/s = the metric
to compare against eval_policy's ~11k/s (M5) and to project Colab gains.

    PYTHONPATH=. python scripts/bench_gpu_eval.py --device mps --steps 400 \\
        --batches 64 128 256 512 1024
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from alphatrain.evaluate import load_model
from alphatrain.gpu_eval_engine import (
    GpuGames, choose_moves, NCELL, label_components_sv, build_observation_t,
)
from scripts.eval_policy_gpu import _init_slot


@torch.no_grad()
def bench(net, dev, dtype, batch, steps, warmup=30):
    st = GpuGames(batch, dev)
    for s in range(batch):
        _init_slot(st, s, 700000 + s)
    st.alive[:] = True

    def one_step():
        labels = label_components_sv(st.boards)
        obs = build_observation_t(st.boards, st.next_pos, st.next_col,
                                  st.n_next, labels=labels)
        out = net(obs.to(dtype))
        logits = (out[0] if isinstance(out, tuple) else out).float()
        moves, has = choose_moves(st.boards, logits, labels=labels)
        st.alive &= has
        mv = moves.clamp(min=0)
        st.step(mv // NCELL, mv % NCELL)

    for _ in range(warmup):
        one_step()
    if dev.type == 'mps':
        torch.mps.synchronize()
    elif dev.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    if dev.type == 'mps':
        torch.mps.synchronize()
    elif dev.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    alive = int(st.alive.sum())
    return steps / dt, alive * steps / dt, dt / steps * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3f.pt')
    p.add_argument('--device', default='mps')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--batches', type=int, nargs='+',
                   default=[64, 128, 256, 512, 1024])
    a = p.parse_args()
    dev = torch.device(a.device)
    net, _ = load_model(a.model, dev, fp16=(dev.type != 'cpu'))
    net.train(False)
    dtype = next(net.parameters()).dtype
    print(f"bench GPU eval engine on {a.device}, {a.steps} steps/batch\n")
    print(f"{'batch':>6} {'steps/s':>9} {'evals/s':>10} {'ms/step':>9}")
    print('-' * 38)
    for b in a.batches:
        try:
            sps, eps, msps = bench(net, dev, dtype, b, a.steps)
            print(f"{b:>6} {sps:>9.1f} {eps:>10.0f} {msps:>9.2f}", flush=True)
        except RuntimeError as e:
            print(f"{b:>6}  FAILED: {str(e)[:50]}", flush=True)
            break
    print(f"\n(eval_policy CPU baseline on M5 ~= 11,000 evals/s)")


if __name__ == '__main__':
    main()
