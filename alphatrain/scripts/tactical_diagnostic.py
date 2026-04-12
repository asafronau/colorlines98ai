"""Tactical Diagnostic: verify the model's policy still matches expert moves.

Takes N expert positions and checks if the model's top-1 policy prediction
matches the expert's top-1 move. This proves the "Tactical DNA" is preserved
in the weights before dropping expert data.

Usage:
    python -m alphatrain.scripts.tactical_diagnostic \
        --model alphatrain/data/pillar2n_best.pt \
        --expert-tensor alphatrain/data/expert_v2_pairwise_g095_surv1.0_ms30.pt \
        --n 1000
"""

import argparse
import numpy as np
import torch
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--expert-tensor', required=True)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = args.device
    print(f"Loading model from {args.model}...", flush=True)
    from alphatrain.evaluate import load_model
    net, max_score = load_model(args.model, device)
    net.train(False)

    print(f"Loading expert tensor from {args.expert_tensor}...", flush=True)
    data = torch.load(args.expert_tensor, weights_only=True, map_location='cpu')
    N = data['boards'].shape[0]
    n_sample = min(args.n, N)

    # Sample random expert positions
    idx = torch.randperm(N)[:n_sample]
    boards = data['boards'][idx]
    pol_indices = data['pol_indices'][idx]  # (n, 5) top-5 move indices
    pol_values = data['pol_values'][idx]    # (n, 5) top-5 move probs
    next_pos = data['next_pos'][idx]
    next_col = data['next_col'][idx]
    n_next = data['n_next'][idx]
    del data

    # Expert top-1 move for each position
    expert_top1 = pol_indices[:, 0].numpy()  # highest-prob move

    # Build observations and get model predictions
    from alphatrain.dataset import TensorDatasetGPU, NUM_MOVES
    print(f"Building observations for {n_sample} positions...", flush=True)

    # Use a minimal dataset just for obs building
    from alphatrain.observation import build_observation
    top1_match = 0
    top5_match = 0
    top10_match = 0

    batch_size = 64
    for start in range(0, n_sample, batch_size):
        end = min(start + batch_size, n_sample)
        batch_boards = boards[start:end].numpy()
        batch_np = next_pos[start:end].numpy()
        batch_nc = next_col[start:end].numpy()
        batch_nn = n_next[start:end].numpy()
        batch_expert = expert_top1[start:end]

        obs_list = []
        nr = np.zeros(3, dtype=np.int32)
        nc = np.zeros(3, dtype=np.int32)
        ncol = np.zeros(3, dtype=np.int32)
        for i in range(end - start):
            nn = int(batch_nn[i])
            for j in range(min(nn, 3)):
                nr[j] = batch_np[i, j, 0]
                nc[j] = batch_np[i, j, 1]
                ncol[j] = batch_nc[i, j]
            obs = build_observation(batch_boards[i], nr, nc, ncol, nn)
            obs_list.append(obs)

        obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(device)
        with torch.inference_mode():
            pol_logits, _ = net(obs_t)

        # Model's top-K predictions
        _, model_topk = pol_logits.topk(10, dim=1)
        model_topk = model_topk.cpu().numpy()

        for i in range(end - start):
            expert_move = batch_expert[i]
            if model_topk[i, 0] == expert_move:
                top1_match += 1
            if expert_move in model_topk[i, :5]:
                top5_match += 1
            if expert_move in model_topk[i, :10]:
                top10_match += 1

    print(f"\n{'='*60}", flush=True)
    print(f"Tactical Diagnostic ({n_sample} expert positions)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Model top-1 matches expert top-1: {top1_match}/{n_sample} "
          f"({100*top1_match/n_sample:.1f}%)", flush=True)
    print(f"  Expert top-1 in model top-5:      {top5_match}/{n_sample} "
          f"({100*top5_match/n_sample:.1f}%)", flush=True)
    print(f"  Expert top-1 in model top-10:     {top10_match}/{n_sample} "
          f"({100*top10_match/n_sample:.1f}%)", flush=True)
    print(f"\nNote: >30% top-1 match is good (6561 possible moves).", flush=True)
    print(f"These numbers are the BASELINE before pure self-play training.", flush=True)
    print(f"After training, re-run to verify tactical DNA is preserved.", flush=True)


if __name__ == '__main__':
    main()
