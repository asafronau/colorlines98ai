"""Phase 2 v4 — DAgger with the full ChatGPT recipe (2026-05-15):

  1. Margin-filtered oracle anchors. Only train on anchors where the oracle's
     best move clearly beats policy_top_1 (margin >= threshold). At K=32,
     Δcap_rate < 0.15 is mostly noise.
  2. Margin-weighted oracle loss. Even within filtered anchors, weight by
     margin so high-confidence corrections dominate.
  3. Broad V12 distillation anchors. Sample N states from V12 self-play +
     crisis JSONs (no oracle labels needed). Forward-KL distill loss applies
     to these too — the trust region now covers the full state distribution.
  4. Conservative LR (default 1e-6). Trust-region intent.
  5. Drift audit at end: on broad anchors, report top-1 agreement with
     pillar2z, KL divergence, entropy change.

Usage:
    python -m alphatrain.scripts.phase2_dagger_v4 \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --oracle alphatrain/data/phase1_oracle.pt \\
        --crisis-dir data/crisis_v12 --selfplay-dir data/selfplay_v12 \\
        --broad-anchors 10000 \\
        --margin-threshold 0.15 \\
        --distill-weight 1.0 \\
        --epochs 10 --batch-size 128 --lr 1e-6 \\
        --tau 5.0 --metric cap_rate \\
        --device mps \\
        --out alphatrain/data/policy_dagger_v4.pt
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
    """Sample N states from V12 JSONs (no oracle labels needed)."""
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
    p.add_argument('--broad-anchors', type=int, default=10000,
                   help='Number of broad V12 states for distillation.')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-6,
                   help='Conservative LR for trust-region fine-tune.')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--tau', type=float, default=5.0)
    p.add_argument('--metric', choices=['cap_rate', 'mean_turns'],
                   default='cap_rate')
    p.add_argument('--margin-threshold', type=float, default=0.15,
                   help='Skip oracle anchors where '
                        'best_cap_rate - policy_top1_cap_rate < this. '
                        'K=32 noise SE ~0.09, so 0.15 is "above noise".')
    p.add_argument('--distill-weight', type=float, default=1.0)
    p.add_argument('--freeze-backbone', action='store_true',
                   help='Freeze stem+blocks+backbone_bn; train only policy head '
                        '(few params, much smaller trust region).')
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
    # IMPORTANT: keep net in inference mode for BatchNorm. BN running stats
    # update on every forward in train mode regardless of requires_grad, which
    # causes silent drift indistinguishable from learning. train(False) freezes
    # BN stats; gradient updates on requires_grad=True params still work.
    net.train(False)

    if args.freeze_backbone:
        n_frozen = 0; n_train = 0
        for name, p in net.named_parameters():
            if name.startswith('policy_'):
                p.requires_grad = True
                n_train += p.numel()
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        print(f"  freeze_backbone: train {n_train:,} head params, "
              f"freeze {n_frozen:,} backbone params", flush=True)

    print(f"Loading frozen reference policy...", flush=True)
    ref_net, _ = load_model(args.model, device, fp16=False, jit_trace=False)
    for pp in ref_net.parameters():
        pp.requires_grad = False
    ref_net.train(False)

    # ── Load oracle anchors, filter by margin ──
    print(f"\nLoading oracle {args.oracle}...", flush=True)
    data = torch.load(args.oracle, weights_only=False)
    results = data['results']
    print(f"  total anchors: {len(results)}", flush=True)

    print(f"\nBuilding oracle training examples "
          f"(margin >= {args.margin_threshold})...", flush=True)
    o_obs = []; o_mv = []; o_tg = []; o_w = []
    o_val_mask = []
    n_skip_fewmoves = 0
    n_skip_lowmargin = 0
    for r in results:
        pm = r['per_move']
        if len(pm) < 2:
            n_skip_fewmoves += 1
            continue
        sorted_moves = sorted(pm.items(), key=lambda kv: kv[1]['rank'])[:6]
        K = len(sorted_moves)
        if args.metric == 'cap_rate':
            qs = np.array([mv['cap_rate'] for _, mv in sorted_moves])
        else:
            qs = np.array([mv['mean_turns'] / 300.0
                           for _, mv in sorted_moves])

        # Margin filter: oracle-best vs policy-top-1 (rank 1 sorted).
        policy_top1_q = qs[0]
        best_q = qs.max()
        margin = best_q - policy_top1_q
        if margin < args.margin_threshold:
            n_skip_lowmargin += 1
            continue

        # Soft target via softmax(tau * q)
        z = args.tau * qs
        z = z - z.max()
        ex = np.exp(z)
        tgt = ex / ex.sum()

        # Observation
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

        mvs = np.full(6, -1, dtype=np.int64)
        tgts = np.zeros(6, dtype=np.float32)
        for k_, (mv, _) in enumerate(sorted_moves):
            mvs[k_] = int(mv)
            tgts[k_] = tgt[k_]

        o_obs.append(obs)
        o_mv.append(mvs)
        o_tg.append(tgts)
        o_w.append(margin)  # weight by margin: surgical correction strength
        o_val_mask.append(
            board_hash_to_bucket(r['anchor_board']) == args.val_bucket)

    n_oracle = len(o_obs)
    print(f"  kept: {n_oracle}  skipped: {n_skip_lowmargin} low-margin, "
          f"{n_skip_fewmoves} few-moves", flush=True)
    if n_oracle == 0:
        print(f"ERROR: 0 oracle anchors after margin filter "
              f"(threshold={args.margin_threshold}). Try lower threshold.",
              flush=True)
        return

    o_obs_t = torch.from_numpy(np.stack(o_obs)).to(device, dtype=torch.float32)
    o_mv_t = torch.from_numpy(np.stack(o_mv)).to(device, dtype=torch.long)
    o_tg_t = torch.from_numpy(np.stack(o_tg)).to(device, dtype=torch.float32)
    o_w_t = torch.from_numpy(np.array(o_w, dtype=np.float32)).to(device)
    o_val_mask = np.array(o_val_mask)
    o_train = np.nonzero(~o_val_mask)[0]
    o_val = np.nonzero(o_val_mask)[0]
    print(f"  oracle train: {len(o_train)}  val: {len(o_val)}", flush=True)
    print(f"  margin stats: mean={np.array(o_w).mean():.3f} "
          f"median={np.median(o_w):.3f} "
          f"max={np.array(o_w).max():.3f}", flush=True)

    # ── Sample broad anchors from V12 JSONs ──
    print(f"\nSampling {args.broad_anchors} broad V12 anchors...",
          flush=True)
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

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                   weight_decay=args.weight_decay)

    def compute_oracle_loss(idxs):
        """Margin-weighted oracle KL on the top-K candidate moves."""
        it = torch.as_tensor(idxs, device=device, dtype=torch.long)
        ot = o_obs_t.index_select(0, it)
        mv = o_mv_t.index_select(0, it)
        tg = o_tg_t.index_select(0, it)
        wts = o_w_t.index_select(0, it)
        logits = net(ot)
        if isinstance(logits, tuple):
            logits = logits[0]
        log_probs = F.log_softmax(logits, dim=1)
        mv_safe = mv.clamp_min(0)
        gathered = torch.gather(log_probs, 1, mv_safe)
        mask = (mv >= 0)
        gathered = gathered.masked_fill(~mask, -1e9)
        gathered_norm = gathered - torch.logsumexp(gathered, dim=1,
                                                    keepdim=True)
        gathered_norm = gathered_norm.masked_fill(~mask, 0.0)
        per_anchor = -(tg * gathered_norm).sum(dim=1)
        # Margin-weighted: divide by mean weight to keep gradient scale stable
        w_norm = wts / wts.mean().clamp_min(1e-6)
        return (per_anchor * w_norm).mean(), logits, log_probs

    def compute_distill_loss(logits, log_probs, obs):
        with torch.no_grad():
            ref_logits = ref_net(obs)
            if isinstance(ref_logits, tuple):
                ref_logits = ref_logits[0]
            ref_log_probs = F.log_softmax(ref_logits, dim=1)
            ref_probs = ref_log_probs.exp()
        # Forward KL: KL(p_ref || p_new) -- penalize dropping ref's mass.
        return (ref_probs * (ref_log_probs - log_probs)).sum(dim=1).mean()

    print(f"\n=== Training {args.epochs} epochs "
          f"(oracle_n={len(o_train)} broad_n={n_broad} lr={args.lr:.1e} "
          f"λ={args.distill_weight}) ===", flush=True)
    best_val = float('inf')
    t0 = time.time()
    rng_np = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        # Stay in train(False) so BN running stats stay frozen (see comment
        # at load_model). Gradient updates on requires_grad=True params still
        # work in eval mode.
        # Each epoch: shuffle oracle indices; we'll do same number of broad
        # batches as oracle batches.
        o_perm = rng_np.permutation(o_train)
        b_perm = rng_np.permutation(n_broad)

        n_batches = max(1, len(o_perm) // args.batch_size)
        running_o = 0.0; running_d_o = 0.0; running_d_b = 0.0; running_n = 0
        for bi in range(n_batches):
            o_batch = o_perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            # Broad batch indices (cycle through b_perm)
            start = (bi * args.batch_size) % n_broad
            end = min(start + args.batch_size, n_broad)
            b_batch = b_perm[start:end]
            if len(b_batch) < args.batch_size:
                # wrap
                need = args.batch_size - len(b_batch)
                b_batch = np.concatenate([b_batch, b_perm[:need]])

            # Oracle forward
            o_loss, o_logits, o_logprobs = compute_oracle_loss(o_batch)
            # Distillation on oracle observations
            d_o_loss = compute_distill_loss(
                o_logits, o_logprobs,
                o_obs_t.index_select(
                    0, torch.as_tensor(o_batch, device=device, dtype=torch.long)))
            # Distillation on broad observations
            b_obs_batch = b_obs_t.index_select(
                0, torch.as_tensor(b_batch, device=device, dtype=torch.long))
            b_logits = net(b_obs_batch)
            if isinstance(b_logits, tuple):
                b_logits = b_logits[0]
            b_logprobs = F.log_softmax(b_logits, dim=1)
            d_b_loss = compute_distill_loss(b_logits, b_logprobs, b_obs_batch)

            loss = o_loss + args.distill_weight * (d_o_loss + d_b_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_o += o_loss.item() * len(o_batch)
            running_d_o += d_o_loss.item() * len(o_batch)
            running_d_b += d_b_loss.item() * len(b_batch)
            running_n += len(o_batch)

        # Val: oracle KL + broad drift audit
        net.train(False)
        with torch.inference_mode():
            v_o = 0.0; v_n = 0
            for bi in range(0, len(o_val), args.batch_size):
                vb = o_val[bi:bi + args.batch_size]
                o_loss, o_logits, o_logprobs = compute_oracle_loss(vb)
                v_o += o_loss.item() * len(vb)
                v_n += len(vb)
            val_oracle = v_o / max(v_n, 1)

            # Drift audit on broad anchors (using a sample)
            audit_idx = rng_np.choice(n_broad, min(2000, n_broad), replace=False)
            audit_obs = b_obs_t.index_select(
                0, torch.as_tensor(audit_idx, device=device, dtype=torch.long))
            new_logits = net(audit_obs)
            if isinstance(new_logits, tuple):
                new_logits = new_logits[0]
            new_log = F.log_softmax(new_logits, dim=1)
            ref_logits = ref_net(audit_obs)
            if isinstance(ref_logits, tuple):
                ref_logits = ref_logits[0]
            ref_log = F.log_softmax(ref_logits, dim=1)
            ref_probs = ref_log.exp()
            forward_kl = (ref_probs * (ref_log - new_log)).sum(dim=1).mean().item()
            top1_agree_drift = (new_logits.argmax(dim=1) ==
                                 ref_logits.argmax(dim=1)).float().mean().item()
            new_entropy = -(new_log.exp() * new_log).sum(dim=1).mean().item()
            ref_entropy = -(ref_probs * ref_log).sum(dim=1).mean().item()

        tl_o = running_o / max(running_n, 1)
        tl_do = running_d_o / max(running_n, 1)
        tl_db = running_d_b / max(running_n, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train oracle={tl_o:.4f} dist_oracle={tl_do:.4f} dist_broad={tl_db:.4f}  "
              f"val oracle={val_oracle:.4f}  "
              f"DRIFT fkl={forward_kl:.4f} top1_agree={top1_agree_drift:.3f} "
              f"H_new={new_entropy:.3f} H_ref={ref_entropy:.3f}  "
              f"[{elapsed:.0f}s]", flush=True)

        # Save by val oracle loss
        if val_oracle < best_val:
            best_val = val_oracle
            torch.save({
                'model': net.state_dict(),
                'channels': net.channels,
                'blocks': net.blocks,
                'epoch': epoch + 1,
                'val_oracle_loss': val_oracle,
                'drift_fkl': forward_kl,
                'drift_top1_agree': top1_agree_drift,
                'max_score': 6050.0,
                'args': vars(args),
                'origin': args.model,
            }, args.out)
            print(f"  ** New best, saved to {args.out} **", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f}m. "
          f"Best val oracle={best_val:.4f}", flush=True)


if __name__ == '__main__':
    main()
