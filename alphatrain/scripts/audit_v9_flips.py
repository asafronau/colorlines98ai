"""Audit v9's argmax flips bucketed by reference confidence.

Per ChatGPT 2026-05-15: global top1_agree is misleading if flips are mostly
between near-tied moves. The right metric is flip rate as a function of
ref's confidence margin (logit_top1 - logit_top2). Bucketed flip rates tell
us whether v9 broke important moves or just flipped harmless ties.
"""

import argparse
import json
import glob
import os
from random import Random
import numpy as np
import torch
import torch.nn.functional as F


def sample_broad(dirs, n, rng):
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, 'game_seed*.json'))))
    out = []
    while len(out) < n:
        f = rng.choice(files)
        try:
            with open(f) as fp:
                game = json.load(fp)
        except Exception:
            continue
        moves = game.get('moves', [])
        if not moves:
            continue
        m = moves[rng.randint(0, len(moves) - 1)]
        out.append({
            'board': np.asarray(m['board'], dtype=np.int8),
            'next_balls': [((int(nb['row']), int(nb['col'])), int(nb['color']))
                            for nb in m['next_balls']],
            'num_next': int(m['num_next']),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', default='alphatrain/data/pillar2z_epoch_19.pt')
    p.add_argument('--new', required=True)
    p.add_argument('--n', type=int, default=10000)
    p.add_argument('--device', default='mps')
    args = p.parse_args()

    from alphatrain.evaluate import load_model
    from alphatrain.observation import build_observation

    device = torch.device(args.device)
    fp16 = (args.device != 'cpu')

    print(f"Loading {args.ref}...", flush=True)
    ref_net, _ = load_model(args.ref, device, fp16=fp16, jit_trace=False)
    ref_net.train(False)
    print(f"Loading {args.new}...", flush=True)
    new_net, _ = load_model(args.new, device, fp16=fp16, jit_trace=False)
    new_net.train(False)

    rng = Random(42)
    print(f"Sampling {args.n} broad states...", flush=True)
    states = sample_broad(['data/crisis_v12', 'data/selfplay_v12'], args.n, rng)
    print(f"  got {len(states)}", flush=True)

    obs_list = []
    for s in states:
        nb_pos = np.zeros((3, 2), dtype=np.int8)
        nb_col = np.zeros(3, dtype=np.int8)
        for k, item in enumerate(s['next_balls'][:3]):
            pos, col = item[0], item[1]
            nb_pos[k, 0] = pos[0]; nb_pos[k, 1] = pos[1]
            nb_col[k] = col
        obs_list.append(build_observation(
            np.asarray(s['board'], dtype=np.int8),
            nb_pos[:, 0].astype(np.intp), nb_pos[:, 1].astype(np.intp),
            nb_col.astype(np.intp), int(s['num_next'])))
    obs = torch.from_numpy(np.stack(obs_list)).to(
        device, dtype=torch.float16 if fp16 else torch.float32)
    n = obs.shape[0]

    ref_top1 = torch.zeros(n, dtype=torch.long, device=device)
    new_top1 = torch.zeros(n, dtype=torch.long, device=device)
    ref_margin = torch.zeros(n, dtype=torch.float32, device=device)
    ref_p_top1 = torch.zeros(n, dtype=torch.float32, device=device)
    new_p_on_ref_top1 = torch.zeros(n, dtype=torch.float32, device=device)

    chunk = 512
    with torch.inference_mode():
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            ref_logits = ref_net(obs[i:j])
            new_logits = new_net(obs[i:j])
            if isinstance(ref_logits, tuple): ref_logits = ref_logits[0]
            if isinstance(new_logits, tuple): new_logits = new_logits[0]
            ref_log = F.log_softmax(ref_logits.float(), dim=1)
            new_log = F.log_softmax(new_logits.float(), dim=1)
            ref_probs = ref_log.exp()
            new_probs = new_log.exp()
            top2_v, top2_i = ref_logits.float().topk(2, dim=1)
            ref_top1[i:j] = top2_i[:, 0]
            ref_margin[i:j] = (top2_v[:, 0] - top2_v[:, 1])
            ref_p_top1[i:j] = ref_probs.gather(1, top2_i[:, :1]).squeeze(1)
            new_top1[i:j] = new_logits.argmax(dim=1)
            new_p_on_ref_top1[i:j] = new_probs.gather(
                1, top2_i[:, :1]).squeeze(1)

    flips = (new_top1 != ref_top1).cpu().numpy()
    margin = ref_margin.cpu().numpy()
    ref_p = ref_p_top1.cpu().numpy()
    new_p_on_ref = new_p_on_ref_top1.cpu().numpy()

    print(f"\n=== Overall ===", flush=True)
    print(f"top1_agree: {100*(1-flips.mean()):.1f}%   "
          f"flip_rate: {100*flips.mean():.1f}%", flush=True)

    print(f"\n=== Flip rate by ref logit margin (top1 - top2) ===", flush=True)
    buckets = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, np.inf)]
    for lo, hi in buckets:
        mask = (margin >= lo) & (margin < hi)
        n_b = mask.sum()
        if n_b == 0: continue
        flip_b = flips[mask].mean()
        print(f"  margin [{lo:.1f}, {hi:.1f}): n={n_b:5d}  "
              f"flip_rate={100*flip_b:5.1f}%", flush=True)

    print(f"\n=== Flip rate by ref top-1 probability ===", flush=True)
    buckets_p = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in buckets_p:
        mask = (ref_p >= lo) & (ref_p < hi)
        n_b = mask.sum()
        if n_b == 0: continue
        flip_b = flips[mask].mean()
        new_p_b = new_p_on_ref[mask].mean()
        print(f"  p_top1 [{lo:.2f}, {hi:.2f}): n={n_b:5d}  "
              f"flip_rate={100*flip_b:5.1f}%  "
              f"new_mass_on_ref_top1={new_p_b:.3f}", flush=True)

    print(f"\n=== Interpretation ===", flush=True)
    print(f"If high-margin (margin>=2) flip rate is low (<5%), "
          f"v9 preserves the 'important' decisions.", flush=True)
    print(f"If low-margin flips dominate, they're between near-tied moves "
          f"and may be harmless.", flush=True)


if __name__ == '__main__':
    main()
