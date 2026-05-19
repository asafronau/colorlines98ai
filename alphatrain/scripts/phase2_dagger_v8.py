"""Phase 2 v9 — DAgger with symmetric KL distillation (v8 hard-CE failed).

v8 failed because hard CE on ref's argmax only constrains 1 of 6,561 logits
per state — the rest drift freely and eventually flip argmaxes. v9 uses
symmetric KL = forward_KL(ref||new) + reverse_KL(new||ref) which constrains
ALL logits in both directions.

v9 vs v8:
  - Broad: symmetric KL instead of hard CE
  - lr 1e-4 → 1e-5 (v8 was too aggressive)
  - Keep margin filter, frozen backbone, hard oracle CE, entropy guard

Original v8 docstring (broken approach kept for reference):


Lesson from v7: forward KL ≤ 0.05 on broad states is NOT sufficient. v7 had
top1_agree=77% on broad and entropy rose 2.10 → 2.35 — both bad. Game eval
regressed to 4,151. The trust-region metric must be BEHAVIORAL.

This run uses HARD distillation (CE to ref's argmax) on broad states +
HARD oracle CE on high-margin anchors only + entropy guard, with the
backbone frozen.

Acceptance criteria before game eval:
  - broad top-1 agreement >= 95%
  - new entropy not increased above ref + 0.02
  - oracle-best probability improved on margin-filtered anchors
  - ref top-1 probability not collapsed broadly

Usage:
    python -m alphatrain.scripts.phase2_dagger_v8 \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --oracle alphatrain/data/phase1_oracle.pt \\
        --crisis-dir data/crisis_v12 --selfplay-dir data/selfplay_v12 \\
        --broad-anchors 50000 \\
        --margin-threshold 0.15 \\
        --broad-weight 5.0 --entropy-weight 1.0 \\
        --top1-floor 0.95 \\
        --epochs 30 --batch-size 256 --lr 1e-4 \\
        --device mps \\
        --out alphatrain/data/policy_dagger_v8.pt
"""

import argparse
import hashlib
import json
import glob
import os
import time
from random import Random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def board_hash_to_bucket(board_int8, n_buckets=10):
    bs = np.ascontiguousarray(board_int8, dtype=np.int8).tobytes()
    h = hashlib.md5(bs).digest()
    return int.from_bytes(h[:4], 'little') % n_buckets


def sample_broad_anchors(dirs, n, rng):
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, 'game_seed*.json'))))
    if not files:
        return []
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 5:
        attempts += 1
        f = rng.choice(files)
        try:
            with open(f) as fp:
                game = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        moves = game.get('moves', [])
        if not moves:
            continue
        mi = rng.randint(0, len(moves) - 1)
        m = moves[mi]
        out.append({
            'board': np.asarray(m['board'], dtype=np.int8),
            'next_balls': [((int(nb['row']), int(nb['col'])), int(nb['color']))
                            for nb in m['next_balls']],
            'num_next': int(m['num_next']),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--oracle', required=True)
    p.add_argument('--crisis-dir', default='data/crisis_v12')
    p.add_argument('--selfplay-dir', default='data/selfplay_v12')
    p.add_argument('--broad-anchors', type=int, default=50000,
                   help='Broad V12 states for behavior-preservation distill.')
    p.add_argument('--margin-threshold', type=float, default=0.15,
                   help='Keep anchors where best_cap_rate - policy_top1_cap_rate '
                        '>= this. K=32 noise SE ~0.09.')
    p.add_argument('--broad-weight', type=float, default=5.0,
                   help='Weight on hard distillation to ref argmax on broad states.')
    p.add_argument('--entropy-weight', type=float, default=1.0,
                   help='Weight on entropy guard penalty (H_new - H_ref - 0.02)^2.')
    p.add_argument('--entropy-tolerance', type=float, default=0.02,
                   help='Max allowed entropy increase over reference.')
    p.add_argument('--top1-floor', type=float, default=0.95,
                   help='Stop training when broad top-1 agreement falls below this.')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--val-bucket', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='mps')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = Random(args.seed)
    device = torch.device(args.device)

    from alphatrain.evaluate import load_model
    from alphatrain.observation import build_observation

    print(f"Loading trainable policy {args.model}...", flush=True)
    net, _ = load_model(args.model, device, fp16=False, jit_trace=False)
    net.train(False)  # BN running stats frozen

    # Freeze backbone — train only policy head.
    n_frozen = 0; n_train = 0
    for name, pp in net.named_parameters():
        if name.startswith('policy_'):
            pp.requires_grad = True
            n_train += pp.numel()
        else:
            pp.requires_grad = False
            n_frozen += pp.numel()
    print(f"  freeze_backbone: train {n_train:,} head params, "
          f"freeze {n_frozen:,} backbone params", flush=True)

    print(f"Loading frozen reference policy...", flush=True)
    ref_net, _ = load_model(args.model, device, fp16=False, jit_trace=False)
    for pp in ref_net.parameters():
        pp.requires_grad = False
    ref_net.train(False)

    # ── Load oracle anchors, filter by margin, build HARD targets ──
    print(f"\nLoading oracle, filtering by margin >= {args.margin_threshold}...",
          flush=True)
    data = torch.load(args.oracle, weights_only=False)
    results = data['results']

    o_obs = []; o_best = []; o_w = []
    o_val_mask = []
    n_skip = 0
    for r in results:
        pm = r['per_move']
        if len(pm) < 2:
            n_skip += 1
            continue
        sorted_moves = sorted(pm.items(), key=lambda kv: kv[1]['rank'])[:6]
        qs = np.array([mv['cap_rate'] for _, mv in sorted_moves])
        policy_top1_q = qs[0]
        best_idx = int(qs.argmax())
        margin = qs[best_idx] - policy_top1_q
        if margin < args.margin_threshold:
            n_skip += 1
            continue
        best_move = int(sorted_moves[best_idx][0])

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

        o_obs.append(obs)
        o_best.append(best_move)
        o_w.append(margin)
        o_val_mask.append(
            board_hash_to_bucket(r['anchor_board']) == args.val_bucket)

    n_oracle = len(o_obs)
    print(f"  kept: {n_oracle}  skipped: {n_skip}", flush=True)
    if n_oracle == 0:
        print(f"ERROR: 0 oracle anchors", flush=True)
        return

    o_obs_t = torch.from_numpy(np.stack(o_obs)).to(device, dtype=torch.float32)
    o_best_t = torch.from_numpy(np.array(o_best, dtype=np.int64)).to(device)
    o_w_t = torch.from_numpy(np.array(o_w, dtype=np.float32)).to(device)
    o_val_mask = np.array(o_val_mask)
    o_train = np.nonzero(~o_val_mask)[0]
    o_val = np.nonzero(o_val_mask)[0]
    print(f"  oracle train: {len(o_train)}  val: {len(o_val)}", flush=True)

    # ── Broad anchors with PRECOMPUTED ref argmax + ref entropy ──
    print(f"\nSampling {args.broad_anchors} broad V12 anchors...", flush=True)
    broad_states = sample_broad_anchors(
        [args.crisis_dir, args.selfplay_dir], args.broad_anchors, rng)
    print(f"  got {len(broad_states)} broad states", flush=True)

    b_obs = []
    for s in broad_states:
        nb_pos = np.zeros((3, 2), dtype=np.int8)
        nb_col = np.zeros(3, dtype=np.int8)
        for k_, item in enumerate(s['next_balls'][:3]):
            pos, col = item[0], item[1]
            nb_pos[k_, 0] = pos[0]
            nb_pos[k_, 1] = pos[1]
            nb_col[k_] = col
        obs = build_observation(
            np.asarray(s['board'], dtype=np.int8),
            nb_pos[:, 0].astype(np.intp),
            nb_pos[:, 1].astype(np.intp),
            nb_col.astype(np.intp),
            int(s['num_next']))
        b_obs.append(obs)
    b_obs_t = torch.from_numpy(np.stack(b_obs)).to(device, dtype=torch.float32)
    n_broad = len(b_obs)

    # Precompute ref argmax + entropy on broad states (one-time cost).
    print(f"Precomputing ref argmax + entropy on broad states...", flush=True)
    ref_argmax_b = torch.zeros(n_broad, dtype=torch.long, device=device)
    ref_entropy_b = torch.zeros(n_broad, dtype=torch.float32, device=device)
    chunk = 512
    with torch.inference_mode():
        for i in range(0, n_broad, chunk):
            j = min(i + chunk, n_broad)
            x = b_obs_t[i:j]
            logits = ref_net(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            log_probs = F.log_softmax(logits, dim=1)
            ref_argmax_b[i:j] = logits.argmax(dim=1)
            ref_entropy_b[i:j] = -(log_probs.exp() * log_probs).sum(dim=1)
    mean_ref_entropy = ref_entropy_b.mean().item()
    print(f"  ref mean entropy on broad: {mean_ref_entropy:.3f}", flush=True)

    trainable_params = [pp for pp in net.parameters() if pp.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                   weight_decay=args.weight_decay)

    print(f"\n=== Training {args.epochs} epochs "
          f"(oracle_n={len(o_train)} broad_n={n_broad} "
          f"broad_w={args.broad_weight} ent_w={args.entropy_weight} "
          f"lr={args.lr:.1e}) ===", flush=True)
    print(f"Acceptance: broad top1_agree >= {args.top1_floor}, "
          f"H_new - H_ref <= {args.entropy_tolerance}", flush=True)

    best_oracle_prob = -1.0
    rng_np = np.random.default_rng(args.seed)
    t0 = time.time()

    for epoch in range(args.epochs):
        o_perm = rng_np.permutation(o_train)
        b_perm = rng_np.permutation(n_broad)

        # Number of mini-batches per epoch = ceil(broad / batch). Oracle is
        # smaller, we cycle through it.
        n_batches = max(1, n_broad // args.batch_size)
        running_or = 0.0; running_br = 0.0; running_en = 0.0
        running_n_or = 0; running_n_br = 0

        for bi in range(n_batches):
            # Oracle: cycle if needed
            ostart = (bi * args.batch_size) % max(len(o_perm), 1)
            o_batch = o_perm[ostart:ostart + args.batch_size]
            if len(o_batch) < min(args.batch_size, len(o_perm)):
                o_batch = np.concatenate(
                    [o_batch, o_perm[:args.batch_size - len(o_batch)]])

            # Broad
            bstart = bi * args.batch_size
            b_batch = b_perm[bstart:bstart + args.batch_size]
            if len(b_batch) == 0:
                continue

            # ── Oracle: hard CE on best move ──
            o_idx = torch.as_tensor(o_batch, device=device, dtype=torch.long)
            o_x = o_obs_t.index_select(0, o_idx)
            o_logits = net(o_x)
            if isinstance(o_logits, tuple):
                o_logits = o_logits[0]
            o_log_probs = F.log_softmax(o_logits, dim=1)
            best = o_best_t.index_select(0, o_idx)
            wts = o_w_t.index_select(0, o_idx)
            wts_norm = wts / wts.mean().clamp_min(1e-6)
            ce_per = -o_log_probs.gather(1, best.unsqueeze(1)).squeeze(1)
            oracle_loss = (ce_per * wts_norm).mean()

            # ── Broad: symmetric KL to ref full distribution ──
            b_idx = torch.as_tensor(b_batch, device=device, dtype=torch.long)
            b_x = b_obs_t.index_select(0, b_idx)
            b_logits = net(b_x)
            if isinstance(b_logits, tuple):
                b_logits = b_logits[0]
            b_log_probs = F.log_softmax(b_logits, dim=1)
            b_probs = b_log_probs.exp()
            with torch.no_grad():
                ref_logits_b = ref_net(b_x)
                if isinstance(ref_logits_b, tuple):
                    ref_logits_b = ref_logits_b[0]
                ref_log_b = F.log_softmax(ref_logits_b, dim=1)
                ref_probs_b = ref_log_b.exp()
            ref_top1 = ref_argmax_b.index_select(0, b_idx)
            ref_H = ref_entropy_b.index_select(0, b_idx)
            # forward KL(ref||new) — covers ref's mass
            fkl = (ref_probs_b * (ref_log_b - b_log_probs)).sum(dim=1)
            # reverse KL(new||ref) — penalizes new mass where ref had none
            rkl = (b_probs * (b_log_probs - ref_log_b)).sum(dim=1)
            broad_loss = (fkl + rkl).mean()

            # ── Entropy guard ──
            new_H = -(b_log_probs.exp() * b_log_probs).sum(dim=1)
            entropy_excess = F.relu(new_H - ref_H - args.entropy_tolerance)
            entropy_loss = (entropy_excess ** 2).mean()

            loss = (oracle_loss
                    + args.broad_weight * broad_loss
                    + args.entropy_weight * entropy_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_or += oracle_loss.item() * len(o_batch)
            running_br += broad_loss.item() * len(b_batch)
            running_en += entropy_loss.item() * len(b_batch)
            running_n_or += len(o_batch)
            running_n_br += len(b_batch)

        # ── Audit on a sample of broad anchors ──
        with torch.inference_mode():
            audit_n = min(2000, n_broad)
            audit_idx = rng_np.choice(n_broad, audit_n, replace=False)
            audit_idx_t = torch.as_tensor(audit_idx, device=device,
                                            dtype=torch.long)
            audit_obs = b_obs_t.index_select(0, audit_idx_t)
            new_logits = net(audit_obs)
            if isinstance(new_logits, tuple):
                new_logits = new_logits[0]
            new_log = F.log_softmax(new_logits, dim=1)
            new_probs = new_log.exp()
            new_argmax = new_logits.argmax(dim=1)
            ref_top1 = ref_argmax_b.index_select(0, audit_idx_t)
            top1_agree = (new_argmax == ref_top1).float().mean().item()
            new_entropy = -(new_probs * new_log).sum(dim=1).mean().item()
            ref_entropy = ref_entropy_b.index_select(0, audit_idx_t).mean().item()
            # Probability the new policy assigns to ref's top-1 action
            mass_on_ref_top1 = new_probs.gather(
                1, ref_top1.unsqueeze(1)).mean().item()

            # Oracle progress: probability assigned to oracle best move
            oracle_logits = net(o_obs_t)
            if isinstance(oracle_logits, tuple):
                oracle_logits = oracle_logits[0]
            oracle_log = F.log_softmax(oracle_logits, dim=1)
            oracle_prob_on_best = oracle_log.gather(
                1, o_best_t.unsqueeze(1)).squeeze(1).exp().mean().item()

        tl_o = running_or / max(running_n_or, 1)
        tl_b = running_br / max(running_n_br, 1)
        tl_e = running_en / max(running_n_br, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"oracle={tl_o:.4f} broad={tl_b:.4f} ent={tl_e:.4f}  "
              f"DRIFT top1_agree={top1_agree:.3f} H_new={new_entropy:.3f} "
              f"H_ref={ref_entropy:.3f} mass_on_ref_top1={mass_on_ref_top1:.3f}  "
              f"oracle_best_prob={oracle_prob_on_best:.3f}  "
              f"[{elapsed:.0f}s]", flush=True)

        # Acceptance check + save
        accept = (top1_agree >= args.top1_floor
                  and new_entropy <= ref_entropy + args.entropy_tolerance)
        if accept and oracle_prob_on_best > best_oracle_prob:
            best_oracle_prob = oracle_prob_on_best
            torch.save({
                'model': net.state_dict(),
                'channels': net.channels,
                'blocks': net.blocks,
                'epoch': epoch + 1,
                'top1_agree': top1_agree,
                'new_entropy': new_entropy,
                'ref_entropy': ref_entropy,
                'oracle_best_prob': oracle_prob_on_best,
                'max_score': 6050.0,
                'args': vars(args),
                'origin': args.model,
            }, args.out)
            print(f"  ** New best (accepted), oracle_best_prob="
                  f"{oracle_prob_on_best:.3f}, saved **", flush=True)

        # Early stop if broad behavior breaks
        if top1_agree < args.top1_floor - 0.05:
            print(f"  ** Top1 agreement {top1_agree:.3f} fell too far below "
                  f"floor {args.top1_floor}. Stopping early. **", flush=True)
            break

    print(f"\nDone in {(time.time()-t0)/60:.1f}m. "
          f"Best accepted oracle_best_prob={best_oracle_prob:.3f}", flush=True)
    if best_oracle_prob < 0:
        print(f"No accepted checkpoint — try lowering broad_weight, "
              f"raising margin_threshold, or starting with smaller lr.",
              flush=True)


if __name__ == '__main__':
    main()
