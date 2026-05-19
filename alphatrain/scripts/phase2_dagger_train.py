"""Phase 2 — DAgger policy distillation from oracle labels.

Loads per-anchor quality vectors from phase1_oracle.pt and fine-tunes the
PolicyNet via KL divergence to soft target distributions over the top-K
oracle-labeled moves at each anchor.

Defaults:
  - target: softmax(tau * cap_rate) over top-6 moves, tau=5
  - loss: KL divergence on the 6 labeled actions only
  - optimizer: AdamW lr=1e-5 wd=1e-4 (fine-tune, not from scratch)
  - epochs: 10
  - val split: 10% of anchors held out, save best by val loss
  - starting point: pillar2z_epoch_19.pt

After: run standard policy-only eval to compare to pillar2z (mean 6952).

Usage:
    python -m alphatrain.scripts.phase2_dagger_train \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --oracle alphatrain/data/phase1_oracle.pt \\
        --epochs 10 --batch-size 128 --lr 1e-5 \\
        --device mps \\
        --out alphatrain/data/policy_dagger_v1.pt
"""

import argparse
import hashlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def board_hash_to_bucket(board_int8, n_buckets=10):
    bs = np.ascontiguousarray(board_int8, dtype=np.int8).tobytes()
    h = hashlib.md5(bs).digest()
    return int.from_bytes(h[:4], 'little') % n_buckets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   help='Starting policy checkpoint (e.g., pillar2z_epoch_19.pt).')
    p.add_argument('--oracle', required=True,
                   help='Phase 1 oracle data (phase1_oracle.pt).')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-5,
                   help='Low default — fine-tuning, not from scratch.')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--tau', type=float, default=5.0,
                   help='Sharpness of target distribution over top-K moves. '
                        'target_i = softmax(tau * cap_rate_i). '
                        'tau=5 -> 7%% cap_rate gap = 1.4x weight ratio.')
    p.add_argument('--metric', choices=['cap_rate', 'mean_turns', 'mixed'],
                   default='cap_rate',
                   help='Quality metric to build soft target from.')
    p.add_argument('--val-bucket', type=int, default=0,
                   help='hash-mod-10 bucket held out for val.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='mps')
    p.add_argument('--distill-weight', type=float, default=1.0,
                   help='Weight on KL distillation from original policy '
                        '(regularizer to prevent catastrophic forgetting). '
                        'Total loss = oracle_KL + distill_weight * distill_KL.')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    fp16 = (args.device != 'cpu')

    from alphatrain.evaluate import load_model
    from alphatrain.observation import build_observation

    print(f"Loading starting policy {args.model} (trainable copy)...",
          flush=True)
    net, _ = load_model(args.model, device, fp16=False, jit_trace=False)
    net.train(True)

    # Load a SECOND frozen copy as the distillation reference. Keeps original
    # policy intact; distill_KL pushes new net to stay close to this on the
    # same anchors. Without this regularizer, fine-tuning on 2K anchors blows
    # away pillar2z's knowledge of all other states (catastrophic forgetting).
    print(f"Loading frozen reference policy for distillation...", flush=True)
    ref_net, _ = load_model(args.model, device, fp16=False, jit_trace=False)
    for pp in ref_net.parameters():
        pp.requires_grad = False
    ref_net.train(False)

    print(f"Loading oracle data {args.oracle}...", flush=True)
    data = torch.load(args.oracle, weights_only=False)
    results = data['results']
    n_anchors_total = len(results)
    print(f"  anchors: {n_anchors_total:,}", flush=True)

    # Build training examples: for each anchor with >=2 moves, take top-K
    # moves (sorted by policy prior already), compute target distribution.
    print(f"\nBuilding training examples (target = softmax({args.tau}*{args.metric}))...",
          flush=True)
    obs_list = []
    move_indices = []  # (N, K_max=6) -- pad with -1 if fewer
    targets = []        # (N, K_max=6) -- pad with 0 if fewer
    sources = []
    val_mask = []
    n_skipped = 0

    for r in results:
        pm = r['per_move']
        if len(pm) < 2:
            n_skipped += 1
            continue

        # Sort moves by policy prior (rank field stored)
        sorted_moves = sorted(pm.items(), key=lambda kv: kv[1]['rank'])
        K = len(sorted_moves)
        if K > 6:
            sorted_moves = sorted_moves[:6]
            K = 6

        # Metric values per move
        if args.metric == 'cap_rate':
            qs = np.array([mv['cap_rate'] for _, mv in sorted_moves])
        elif args.metric == 'mean_turns':
            # Normalize to roughly [0, 1] by horizon=300
            qs = np.array([mv['mean_turns'] / 300.0 for _, mv in sorted_moves])
        else:  # mixed
            cap = np.array([mv['cap_rate'] for _, mv in sorted_moves])
            turns = np.array([mv['mean_turns'] / 300.0 for _, mv in sorted_moves])
            qs = 0.7 * cap + 0.3 * turns

        # Softmax target with sharpness tau
        z = args.tau * qs
        z = z - z.max()
        ex = np.exp(z)
        tgt = ex / ex.sum()

        # Build observation
        nb_pos = np.zeros((3, 2), dtype=np.int8)
        nb_col = np.zeros(3, dtype=np.int8)
        for k_, item in enumerate(r['anchor_next_balls'][:3]):
            pos, col = item[0], item[1]
            nb_pos[k_, 0] = pos[0]
            nb_pos[k_, 1] = pos[1]
            nb_col[k_] = col
        obs = build_observation(
            np.asarray(r['anchor_board'], dtype=np.int8),
            nb_pos[:, 0].astype(np.intp),
            nb_pos[:, 1].astype(np.intp),
            nb_col.astype(np.intp),
            int(r['anchor_n_next']))

        # Pad to K_max=6
        mvs = np.full(6, -1, dtype=np.int64)
        tgts = np.zeros(6, dtype=np.float32)
        for k_, (mv, _) in enumerate(sorted_moves):
            mvs[k_] = int(mv)
            tgts[k_] = tgt[k_]

        obs_list.append(obs)
        move_indices.append(mvs)
        targets.append(tgts)
        sources.append(r['source_label'])
        val_mask.append(
            board_hash_to_bucket(r['anchor_board']) == args.val_bucket)

    obs_arr = np.stack(obs_list)
    move_arr = np.stack(move_indices)
    tgt_arr = np.stack(targets)
    val_mask = np.array(val_mask)
    N = obs_arr.shape[0]
    print(f"  built {N:,} (skipped {n_skipped})", flush=True)
    n_train = int((~val_mask).sum())
    n_val = int(val_mask.sum())
    print(f"  train: {n_train:,}  val: {n_val:,}  "
          f"(split by anchor_board hash mod 10 == {args.val_bucket})",
          flush=True)

    obs_t = torch.from_numpy(obs_arr).to(device, dtype=torch.float32)
    move_t = torch.from_numpy(move_arr).to(device, dtype=torch.long)
    tgt_t = torch.from_numpy(tgt_arr).to(device, dtype=torch.float32)
    train_idx = np.nonzero(~val_mask)[0]
    val_idx = np.nonzero(val_mask)[0]

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    def forward_loss(idxs):
        """Compute oracle KL + distillation KL.

        oracle_loss: KL between policy(top-K) and target_dist over the
            K oracle-labeled moves. Teaches the new policy what oracle prefers.

        distill_loss: full KL divergence between new policy(state) and frozen
            reference policy(state) over ALL 6561 actions. Prevents the new
            policy from drifting away from pillar2z's general knowledge.
        """
        idx_t = torch.as_tensor(idxs, device=device, dtype=torch.long)
        ot = obs_t.index_select(0, idx_t)
        mv = move_t.index_select(0, idx_t)
        tg = tgt_t.index_select(0, idx_t)

        logits = net(ot)
        if isinstance(logits, tuple):
            logits = logits[0]
        log_probs_full = F.log_softmax(logits, dim=1)  # (B, 6561)

        with torch.no_grad():
            ref_logits = ref_net(ot)
            if isinstance(ref_logits, tuple):
                ref_logits = ref_logits[0]
            ref_log_probs = F.log_softmax(ref_logits, dim=1)
            ref_probs = ref_log_probs.exp()

        # Distillation: KL(p_new || p_ref) ≈ sum p_new * (log p_new - log p_ref)
        # Use forward KL: KL(p_ref || p_new) = sum p_ref * (log p_ref - log p_new)
        # Forward KL pulls p_new to cover p_ref's mass — more conservative.
        distill_loss = (ref_probs * (ref_log_probs - log_probs_full)).sum(dim=1).mean()

        # Oracle loss on top-K candidate moves.
        mv_safe = mv.clamp_min(0)
        gathered_log = torch.gather(log_probs_full, 1, mv_safe)  # (B, 6)
        mask = (mv >= 0)
        # Renormalize log_probs over the K candidate moves only:
        gathered_log = gathered_log.masked_fill(~mask, -1e9)
        gathered_log_norm = gathered_log - torch.logsumexp(
            gathered_log, dim=1, keepdim=True)
        # Zero out padding contribution (tg=0 there anyway).
        gathered_log_norm = gathered_log_norm.masked_fill(~mask, 0.0)
        oracle_loss = -(tg * gathered_log_norm).sum(dim=1).mean()

        # Top-1 agreement of the new policy with the oracle's best move.
        gathered_for_argmax = gathered_log.detach()
        pred_top1 = gathered_for_argmax.argmax(dim=1)
        tgt_top1 = tg.argmax(dim=1)
        top1_agree = (pred_top1 == tgt_top1).float().mean()

        return oracle_loss, distill_loss, top1_agree

    print(f"\n=== Training {args.epochs} epochs "
          f"(bs={args.batch_size}, lr={args.lr:.1e}, tau={args.tau}) ===",
          flush=True)
    rng = np.random.default_rng(args.seed)
    best_val = float('inf')
    t0 = time.time()

    for epoch in range(args.epochs):
        net.train(True)
        perm = rng.permutation(train_idx)
        running_oracle = 0.0
        running_distill = 0.0
        running_acc = 0.0
        running_n = 0
        for bi in range(0, len(perm), args.batch_size):
            batch = perm[bi:bi + args.batch_size]
            o_loss, d_loss, acc = forward_loss(batch)
            loss = o_loss + args.distill_weight * d_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_oracle += o_loss.item() * len(batch)
            running_distill += d_loss.item() * len(batch)
            running_acc += acc.item() * len(batch)
            running_n += len(batch)

        net.train(False)
        with torch.inference_mode():
            v_oracle = 0.0
            v_distill = 0.0
            v_acc = 0.0
            v_n = 0
            for bi in range(0, len(val_idx), args.batch_size):
                batch = val_idx[bi:bi + args.batch_size]
                o_loss, d_loss, acc = forward_loss(batch)
                v_oracle += o_loss.item() * len(batch)
                v_distill += d_loss.item() * len(batch)
                v_acc += acc.item() * len(batch)
                v_n += len(batch)
            val_oracle = v_oracle / max(v_n, 1)
            val_distill = v_distill / max(v_n, 1)
            val_loss = val_oracle + args.distill_weight * val_distill
            val_acc = v_acc / max(v_n, 1)

        tl_o = running_oracle / max(running_n, 1)
        tl_d = running_distill / max(running_n, 1)
        tl = tl_o + args.distill_weight * tl_d
        ta = running_acc / max(running_n, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train oracle={tl_o:.4f} distill={tl_d:.4f} top1={ta:.3f}  "
              f"val oracle={val_oracle:.4f} distill={val_distill:.4f} top1={val_acc:.3f}  "
              f"[{elapsed:.0f}s]", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model': net.state_dict(),  # key 'model' is what load_model expects
                'channels': net.channels,
                'blocks': net.blocks,
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_loss': tl,
                'train_acc': ta,
                'max_score': 6050.0,  # passthrough metadata
                'args': vars(args),
                'origin': args.model,
            }, args.out)
            print(f"  ** New best, saved to {args.out} **", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f}m. Best val_loss={best_val:.4f}",
          flush=True)


if __name__ == '__main__':
    main()
