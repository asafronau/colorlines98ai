"""AlphaTrain Path B training: V12 policy + oracle correction auxiliary.

Forks alphatrain/train.py. Adds a small oracle anchor corpus as an auxiliary
conditional-KL loss with quadratic reliability weighting.

    loss = L_v12 + λ · weighted_mean( KL( softmax(β · cap_rates_top6) ‖
                                            softmax(gather(logits, actions_top6)) ) )

Reliability weight per anchor:
    w = clip( (Δcap_rate − noise_floor) / scale , 0, 1 ) ** 2

Run B (λ=0): identical pipeline to train.py + color aug, without oracle loss.
Runs C, D: same data, λ ∈ {0.05, 0.10}.

Usage (Colab):
    python -m alphatrain.train_path_b \\
        --tensor-file alphatrain/data/selfplay.pt \\
        --oracle-tensor alphatrain/data/phase1_oracle_path_b.pt \\
        --oracle-lambda 0.05 \\
        --amp --compile \\
        --resume alphatrain/data/pillar2y2_epoch_40.pt --warm-start \\
        --color-augment \\
        --epochs 40 --batch-size 32768 --lr 3e-4 --warmup-epochs 2 \\
        --copy-to /content/drive/MyDrive/alphatrain/path_b_C_best.pt \\
        --save-dir /content/checkpoints/path_b_C
"""

from __future__ import annotations

import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.model import AlphaTrainNet, count_parameters
from alphatrain.observation import build_observation


# Sentinel for masked-out softmax slots. Large-negative finite avoids
# NaN from 0*-inf at pad positions while giving softmax probabilities ~0.
NEG_INF = -1e9


def cross_entropy_soft(logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def distillation_loss(logits, soft_targets, blend_alpha=1.0,
                      target_temperature=1.0):
    """V12 policy loss — soft CE on visit distribution, optionally blended
    with hard CE on the argmax and/or sharpened by temperature.

    blend_alpha=1.0, target_temperature=1.0 → original soft CE (V12 baseline).
    """
    if target_temperature != 1.0:
        sharp = soft_targets.pow(1.0 / target_temperature)
        sharp = sharp / sharp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        soft_targets = sharp
    log_probs = F.log_softmax(logits, dim=-1)
    soft = -(soft_targets * log_probs).sum(dim=-1).mean()
    if blend_alpha >= 1.0:
        return soft
    argmax_idx = soft_targets.argmax(dim=-1)
    hard = F.cross_entropy(logits, argmax_idx)
    return blend_alpha * soft + (1.0 - blend_alpha) * hard


def reliability_weight(delta_cap, noise_floor=0.05, scale=0.20):
    """Quadratic ramp: w(Δ) = clip((Δ − floor)/scale, 0, 1)²

      Δ <= 0.05: w = 0       (noise floor; ignored)
      Δ  = 0.10: w = 0.0625   (weak)
      Δ  = 0.15: w = 0.25     (medium)
      Δ >= 0.25: w = 1.0      (dominant)
    """
    return (((delta_cap - noise_floor) / scale).clamp(0.0, 1.0)) ** 2


def oracle_loss(logits, actions, cap_rates, n_moves, delta_cap,
                 beta=10.0, noise_floor=0.05, scale=0.20):
    """Reliability-weighted conditional KL over top-K candidate logits.

    Args:
        logits:     (B, 6561) policy logits from net
        actions:    (B, K)    int64, src*81+tgt; pad value = -1
        cap_rates:  (B, K)    float32, pad value = 0.0
        n_moves:    (B,)      int (2..K), number of valid candidates per anchor
        delta_cap:  (B,)      float32, ≥ 0; max(cap_rate) − rank1(cap_rate)
        beta:       scalar, softmax temperature on cap_rates
        noise_floor, scale: reliability ramp parameters

    Returns:
        loss:    scalar — Σ(w·KL) / max(Σw, 1)
        weights: (B,)  — per-anchor reliability weights (for diagnostics)
    """
    B, K = actions.shape
    arange_k = torch.arange(K, device=actions.device)
    n_long = n_moves.to(torch.long).unsqueeze(1)
    mask = arange_k.unsqueeze(0) < n_long

    safe_actions = actions.clamp(min=0)
    gathered = logits.gather(1, safe_actions)
    neg_inf = torch.tensor(NEG_INF, device=logits.device, dtype=logits.dtype)

    cand_logits = torch.where(mask, gathered, neg_inf)
    score = torch.where(mask, beta * cap_rates,
                         torch.full_like(cap_rates, NEG_INF))
    target = F.softmax(score, dim=-1)
    log_pred = F.log_softmax(cand_logits, dim=-1)

    log_target = torch.log(target.clamp(min=1e-30))
    kl = target * (log_target - log_pred)
    kl = torch.where(mask, kl, torch.zeros_like(kl))
    kl_per_anchor = kl.sum(dim=-1)

    w = reliability_weight(delta_cap, noise_floor, scale)
    denom = w.sum().clamp(min=1.0)
    loss = (w * kl_per_anchor).sum() / denom
    return loss, w


class OracleDataset:
    """In-memory oracle corpus. Pre-builds 18-channel obs on load."""

    def __init__(self, path, device):
        d = torch.load(path, weights_only=False)
        boards = d['boards'].numpy()
        next_pos = d['next_pos'].numpy()
        next_col = d['next_col'].numpy()
        n_next = d['n_next'].numpy()
        N = boards.shape[0]

        obs = np.empty((N, 18, 9, 9), dtype=np.float32)
        for i in range(N):
            nn_i = int(n_next[i])
            obs[i] = build_observation(
                boards[i],
                next_pos[i, :nn_i, 0].astype(np.int64),
                next_pos[i, :nn_i, 1].astype(np.int64),
                next_col[i, :nn_i].astype(np.int64),
                nn_i,
            )

        self.obs = torch.from_numpy(obs).to(device)
        self.actions = d['actions'].to(device)
        self.cap_rates = d['cap_rates'].to(device)
        self.n_moves = d['n_moves'].to(device)
        self.delta_cap = d['delta_cap'].to(device)
        self.N = N
        self.size_mb = (
            self.obs.element_size() * self.obs.numel() / 1e6)

    def split(self, val_frac=0.05, seed=42):
        """Returns (train_idx, val_idx)."""
        g = torch.Generator(device='cpu').manual_seed(seed)
        perm = torch.randperm(self.N, generator=g)
        n_val = max(1, int(self.N * val_frac))
        return (perm[n_val:].to(self.obs.device),
                perm[:n_val].to(self.obs.device))

    def sample(self, indices, batch_size, generator):
        """Sample `batch_size` rows from `indices` (with replacement)."""
        sel = indices[torch.randint(0, indices.shape[0], (batch_size,),
                                      generator=generator,
                                      device=indices.device)]
        return (self.obs[sel], self.actions[sel], self.cap_rates[sel],
                self.n_moves[sel], self.delta_cap[sel])


@torch.no_grad()
def oracle_validate(model, ods, val_idx, device, beta,
                     noise_floor, scale, batch_size=4096,
                     amp_dtype=torch.float32):
    """Compute oracle val metrics over the held-out anchors.

    Returns dict with: kl_all, kl_weighted, top1_all, top1_d10, top1_d15,
    top1_d25, n_val, n_d10, n_d15, n_d25.
    """
    model.train(False)
    use_amp = amp_dtype != torch.float32

    n_val = int(val_idx.shape[0])
    kl_sum = 0.0
    kl_w_sum = 0.0
    w_sum = 0.0
    n_top1 = {'all': 0, 'd10': 0, 'd15': 0, 'd25': 0}
    n_bucket = {'all': 0, 'd10': 0, 'd15': 0, 'd25': 0}

    for start in range(0, n_val, batch_size):
        sel = val_idx[start:start + batch_size]
        obs = ods.obs[sel]
        actions = ods.actions[sel]
        cap_rates = ods.cap_rates[sel]
        n_moves = ods.n_moves[sel]
        delta_cap = ods.delta_cap[sel]
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(obs)
            logits = out[0] if isinstance(out, tuple) else out
        logits = logits.float()

        B, K = actions.shape
        arange_k = torch.arange(K, device=device)
        mask = arange_k.unsqueeze(0) < n_moves.to(torch.long).unsqueeze(1)
        safe_actions = actions.clamp(min=0)
        gathered = logits.gather(1, safe_actions)
        cand_logits = torch.where(
            mask, gathered, torch.full_like(gathered, NEG_INF))
        score = torch.where(mask, beta * cap_rates,
                              torch.full_like(cap_rates, NEG_INF))
        target = F.softmax(score, dim=-1)
        log_pred = F.log_softmax(cand_logits, dim=-1)
        log_target = torch.log(target.clamp(min=1e-30))
        kl = target * (log_target - log_pred)
        kl = torch.where(mask, kl, torch.zeros_like(kl))
        kl_per = kl.sum(dim=-1)
        w = reliability_weight(delta_cap, noise_floor, scale)

        kl_sum += kl_per.sum().item()
        kl_w_sum += (w * kl_per).sum().item()
        w_sum += w.sum().item()

        model_top1 = cand_logits.argmax(dim=-1)
        oracle_top1 = (beta * cap_rates +
                        torch.where(mask, torch.zeros_like(cap_rates),
                                     torch.full_like(cap_rates, NEG_INF))
                        ).argmax(dim=-1)
        agree = (model_top1 == oracle_top1)
        n_top1['all'] += agree.sum().item()
        n_bucket['all'] += B
        for tag, thresh in (('d10', 0.10), ('d15', 0.15), ('d25', 0.25)):
            sel_b = delta_cap >= thresh
            n_top1[tag] += (agree & sel_b).sum().item()
            n_bucket[tag] += sel_b.sum().item()

    def _safe_div(a, b):
        return float(a) / float(b) if b > 0 else 0.0

    return {
        'kl_all': _safe_div(kl_sum, n_val),
        'kl_weighted': _safe_div(kl_w_sum, w_sum),
        'top1_all': _safe_div(n_top1['all'], n_bucket['all']),
        'top1_d10': _safe_div(n_top1['d10'], n_bucket['d10']),
        'top1_d15': _safe_div(n_top1['d15'], n_bucket['d15']),
        'top1_d25': _safe_div(n_top1['d25'], n_bucket['d25']),
        'n_val': n_val,
        'n_d10': n_bucket['d10'],
        'n_d15': n_bucket['d15'],
        'n_d25': n_bucket['d25'],
    }


def train_epoch_path_b(model, loader, optimizer, device, scaler,
                        ods, train_oracle_idx, oracle_args, amp_dtype,
                        log_interval=100, blend_alpha=1.0,
                        target_temperature=1.0, sample_gen=None):
    """One epoch. Each step: V12 batch (+ optional oracle batch concat)."""
    model.train(True)
    use_amp = amp_dtype != torch.float32

    oracle_lambda = oracle_args['lambda']
    use_oracle = (ods is not None) and (oracle_lambda > 0)
    oracle_bs = oracle_args.get('batch_size', 4096)
    beta = oracle_args['beta']
    noise_floor = oracle_args['noise_floor']
    scale = oracle_args['scale']

    total_loss = 0.0
    total_v12 = 0.0
    total_oracle = 0.0
    n = 0
    t0 = time.time()

    for bi, batch in enumerate(loader):
        obs_v, pol_v = batch
        obs_v = obs_v.to(device, non_blocking=True)
        pol_v = pol_v.to(device, non_blocking=True)

        if use_oracle:
            o_obs, o_act, o_cap, o_nm, o_dc = ods.sample(
                train_oracle_idx, oracle_bs, sample_gen)
            B_v = obs_v.shape[0]
            obs_cat = torch.cat([obs_v, o_obs], dim=0)
        else:
            obs_cat = obs_v
            B_v = obs_v.shape[0]

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(obs_cat)
            logits = out[0] if isinstance(out, tuple) else out
            v12_logits = logits[:B_v]
            L_v12 = distillation_loss(
                v12_logits, pol_v, blend_alpha=blend_alpha,
                target_temperature=target_temperature)
            if use_oracle:
                o_logits = logits[B_v:].float()
                L_oracle, _ = oracle_loss(
                    o_logits, o_act, o_cap, o_nm, o_dc,
                    beta=beta, noise_floor=noise_floor, scale=scale)
                loss = L_v12 + oracle_lambda * L_oracle
            else:
                L_oracle = torch.zeros((), device=device)
                loss = L_v12

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_v12 += L_v12.item()
        total_oracle += L_oracle.item()
        n += 1

        if (bi + 1) % log_interval == 0:
            elapsed = time.time() - t0
            sps = (bi + 1) * loader.batch_size / elapsed
            eta = (len(loader) - bi - 1) * loader.batch_size / max(sps, 1)
            avg_oracle = (oracle_lambda * total_oracle / n) if use_oracle else 0.0
            print(f"  [{bi+1}/{len(loader)}] "
                  f"loss={total_loss/n:.4f} "
                  f"(v12={total_v12/n:.4f} "
                  f"+ λ·oracle={avg_oracle:.4f}) "
                  f"{sps:.0f} s/s ETA {eta/60:.0f}m", flush=True)

    return total_loss / n, total_v12 / n, total_oracle / n


@torch.no_grad()
def validate_v12(model, loader, device, amp_dtype=torch.float32):
    model.train(False)
    use_amp = amp_dtype != torch.float32
    total = 0.0
    n = 0
    for obs, pol_tgt in loader:
        obs = obs.to(device, non_blocking=True)
        pol_tgt = pol_tgt.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(obs)
            logits = out[0] if isinstance(out, tuple) else out
            loss = cross_entropy_soft(logits, pol_tgt)
        total += loss.item()
        n += 1
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor-file', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--val-split', type=float, default=0.05)
    p.add_argument('--color-augment', action='store_true')
    p.add_argument('--no-dihedral-augment', action='store_true')
    p.add_argument('--augment-factor', type=int, default=8)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--warm-start', action='store_true')
    p.add_argument('--save-dir', default='checkpoints/path_b')
    p.add_argument('--copy-to', type=str, default=None)
    p.add_argument('--blend-alpha', type=float, default=1.0)
    p.add_argument('--target-temperature', type=float, default=1.0)
    p.add_argument('--oracle-tensor', type=str, default=None,
                   help='Path to phase1_oracle_path_b.pt. Required if '
                        '--oracle-lambda > 0.')
    p.add_argument('--oracle-lambda', type=float, default=0.0,
                   help='Mixing coef. 0=run B (no oracle). 0.05/0.10=C/D.')
    p.add_argument('--oracle-beta', type=float, default=10.0)
    p.add_argument('--oracle-noise-floor', type=float, default=0.05)
    p.add_argument('--oracle-scale', type=float, default=0.20)
    p.add_argument('--oracle-batch-size', type=int, default=4096)
    p.add_argument('--oracle-val-frac', type=float, default=0.05)
    p.add_argument('--oracle-seed', type=int, default=2026)
    args = p.parse_args()

    if args.oracle_lambda > 0 and args.oracle_tensor is None:
        raise SystemExit(
            "ERROR: --oracle-lambda > 0 requires --oracle-tensor. Aborting.")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    train_set, val_set = TensorDatasetGPU.make_train_val_split(
        args.tensor_file,
        val_split=args.val_split,
        augment=not args.no_dihedral_augment,
        color_augment=args.color_augment,
        augment_factor=args.augment_factor,
        device=str(device),
        seed=42,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0,
                              collate_fn=train_set.collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=0,
                            collate_fn=val_set.collate)
    print(f"V12 train={len(train_set):,}  val={len(val_set):,}",
          flush=True)

    ods = None
    train_oracle_idx = None
    val_oracle_idx = None
    if args.oracle_lambda > 0:
        print(f"Loading oracle tensor {args.oracle_tensor}...", flush=True)
        ods = OracleDataset(args.oracle_tensor, device=device)
        train_oracle_idx, val_oracle_idx = ods.split(
            val_frac=args.oracle_val_frac, seed=args.oracle_seed)
        print(f"  oracle anchors: train={int(train_oracle_idx.numel()):,}  "
              f"val={int(val_oracle_idx.numel()):,}  "
              f"(obs on {device}, {ods.size_mb:.0f} MB)", flush=True)
        w_all = reliability_weight(
            ods.delta_cap, args.oracle_noise_floor, args.oracle_scale).cpu()
        print(f"  reliability w: mean {w_all.mean():.4f}  "
              f">0: {(w_all > 0).sum().item()} "
              f"({100*(w_all > 0).float().mean().item():.1f}%)  "
              f"==1: {(w_all >= 0.999).sum().item()}", flush=True)

    model = AlphaTrainNet(num_blocks=args.num_blocks,
                          channels=args.channels).to(device)
    n_params = count_parameters(model)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    print(f"Model: {args.num_blocks}b x {args.channels}ch, "
          f"{n_params:,} params", flush=True)

    start_epoch = 0
    best_val = float('inf')
    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(f"--resume not found: {args.resume}")
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device,
                            weights_only=False)
        state = ckpt['model']
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                     if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in state if k not in filtered]
        model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"  Skipped {len(skipped)} keys (shape/name mismatch)",
                  flush=True)
        if not args.warm_start and 'optimizer' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            best_val = ckpt.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch - 1}", flush=True)
        else:
            print(f"Warm-start from epoch {ckpt.get('epoch', '?')}, fresh "
                  f"optimizer", flush=True)

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("torch.compile enabled", flush=True)

    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(train_params, lr=args.lr,
                                   weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    if args.resume and not args.warm_start and ckpt is not None \
            and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    use_amp = args.amp and device.type == 'cuda'
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = None
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler('cuda')
    else:
        amp_dtype = torch.float32
        scaler = None
    if use_amp:
        print(f"AMP enabled (dtype={amp_dtype})", flush=True)

    oracle_args = dict(
        **{'lambda': args.oracle_lambda},
        beta=args.oracle_beta,
        noise_floor=args.oracle_noise_floor,
        scale=args.oracle_scale,
        batch_size=args.oracle_batch_size,
    )
    sample_gen = torch.Generator(device=device.type).manual_seed(
        args.oracle_seed)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.oracle_lambda > 0:
        print(f"\n=== Path B run with oracle λ={args.oracle_lambda}, "
              f"β={args.oracle_beta}, floor={args.oracle_noise_floor}, "
              f"scale={args.oracle_scale} ===", flush=True)
    else:
        print(f"\n=== Path B run (λ=0, oracle aux DISABLED) ===",
              flush=True)
    print(f"Training {args.epochs} epochs", flush=True)

    t_total = time.time()
    for epoch in range(start_epoch, args.epochs):
        et = time.time()
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.2e})", flush=True)

        tl, v12_l, ora_l = train_epoch_path_b(
            model, train_loader, optimizer, device, scaler,
            ods, train_oracle_idx, oracle_args, amp_dtype,
            blend_alpha=args.blend_alpha,
            target_temperature=args.target_temperature,
            sample_gen=sample_gen)
        vl = validate_v12(model, val_loader, device, amp_dtype=amp_dtype)
        scheduler.step()

        print(f"  Train: loss={tl:.4f}  v12={v12_l:.4f}  "
              f"oracle={ora_l:.4f}", flush=True)
        print(f"  V12 val: loss={vl:.4f} [{time.time()-et:.0f}s]",
              flush=True)

        if args.oracle_lambda > 0:
            om = oracle_validate(
                model, ods, val_oracle_idx, device,
                beta=args.oracle_beta,
                noise_floor=args.oracle_noise_floor,
                scale=args.oracle_scale,
                amp_dtype=amp_dtype)
            print(f"  Oracle val: KL_all={om['kl_all']:.4f}  "
                  f"KL_weighted={om['kl_weighted']:.4f}  "
                  f"top1_all={om['top1_all']:.3f} ({om['n_val']})  "
                  f"top1≥.10={om['top1_d10']:.3f} ({om['n_d10']})  "
                  f"top1≥.15={om['top1_d15']:.3f} ({om['n_d15']})  "
                  f"top1≥.25={om['top1_d25']:.3f} ({om['n_d25']})",
                  flush=True)

        ck = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': vl,
            'best_val_loss': min(best_val, vl),
            'args': vars(args),
            'policy_only': True,
        }
        latest = os.path.join(args.save_dir, 'latest.pt')
        torch.save(ck, latest)
        epoch_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pt')
        torch.save(ck, epoch_path)
        if args.copy_to:
            copy_dir = os.path.dirname(args.copy_to) or '.'
            # Derive run prefix from --save-dir basename so parallel notebooks
            # don't clobber each other's mid-training checkpoints on Drive.
            run_prefix = os.path.basename(args.save_dir.rstrip('/')) or 'path_b'
            shutil.copy2(latest,
                          os.path.join(copy_dir, f'{run_prefix}_latest.pt'))
            shutil.copy2(epoch_path,
                          os.path.join(copy_dir,
                                        f'{run_prefix}_epoch_{epoch+1}.pt'))
        if vl < best_val:
            best_val = vl
            best_path = os.path.join(args.save_dir, 'best.pt')
            torch.save(ck, best_path)
            print(f"  ** New best (val={vl:.4f}) **", flush=True)
            if args.copy_to:
                shutil.copy2(best_path, args.copy_to)

    print(f"\n=== Done: {args.epochs} epochs in "
          f"{(time.time()-t_total)/3600:.1f}h ===", flush=True)
    print(f"Best V12 val: {best_val:.4f}", flush=True)


if __name__ == '__main__':
    main()
