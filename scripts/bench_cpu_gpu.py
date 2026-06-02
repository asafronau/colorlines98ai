"""Measure rollout throughput on a given device (cpu/mps), to evaluate a
CPU+GPU hybrid: run one GPU rollout process AND one CPU-only process in
parallel on disjoint compute (M5 has 18 cores; GPU mining leaves ~17 idle).
Hybrid throughput should be ~ GPU + CPU if they don't contend.

    PYTHONPATH=. python scripts/bench_cpu_gpu.py --device cpu --threads 16
    PYTHONPATH=. python scripts/bench_cpu_gpu.py --device mps
"""
import os
import sys
import time
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.evaluate import load_model
from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
from scripts.batched_rollout import batched_rollout, restore, _decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=0, help='torch CPU threads (0=default)')
    p.add_argument('--batch', type=int, default=0, help='0=auto (cpu 64, gpu 128)')
    p.add_argument('--n-cand', type=int, default=6)
    p.add_argument('--seeds', type=int, default=80)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--tag', default='')
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    fp16 = (dev.type != 'cpu')
    batch = a.batch or (64 if dev.type == 'cpu' else 128)
    net, _ = load_model('alphatrain/data/pillar3b_epoch_20.pt', dev, fp16=fp16)
    dtype = next(net.parameters()).dtype

    game = sorted(glob.glob('alphatrain/data/death_games/death_*.json'))[0]
    d = json.load(open(game))
    fr = d['frames'][len(d['frames']) - 1 - 30]
    anchor = {'board': fr['board'], 'next_balls': fr['next_balls'],
              'score': fr.get('score_before', fr['score']), 'turn': fr['turn']}
    g0 = restore(anchor, 0)
    obs = torch.from_numpy(_build_obs_for_game(g0)).unsqueeze(0).to(dev, dtype)
    with torch.no_grad():
        logits = net(obs)[0].float().cpu().numpy()
    pri = _get_legal_priors_flat(g0.board, logits, 64)
    cand = [_decode(m) for m, _ in sorted(pri.items(), key=lambda x: -x[1])[:a.n_cand]]
    jobs = [(anchor, c, s) for c in cand for s in range(a.seeds)]

    batched_rollout(net, dev, dtype, jobs[:batch], 20, batch)        # warmup
    t = time.perf_counter()
    res = batched_rollout(net, dev, dtype, jobs, a.horizon, batch)
    dt = time.perf_counter() - t
    turns = sum(x['turns'] for x in res)
    pre = f"[{a.tag}] " if a.tag else ""
    print(f"{pre}device={a.device} threads={torch.get_num_threads()} "
          f"batch={batch}: {turns} turns in {dt:.1f}s = {turns/dt:.0f} turns/s",
          flush=True)


if __name__ == '__main__':
    main()
