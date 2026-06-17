"""Self-play TRUNK distillation with the completed-Q improvement target (the Gumbel fix).

Replaces visit-distillation (which regressed via prior-domination — see
docs/gumbel_target_spec_for_review.md). The dataset (GumbelDatasetGPU) yields, per state,
a dense improvement target = softmax(clean prior + gated advantage/τ) and a dense anchor =
softmax(clean prior), plus a per-state weight 1+γ·1[trustworthy correction]. The advantage
is gated to the ~5% of states with a well-visited, value-separated better move; on the
other ~95% target == prior exactly, so they're simply held at the prior.

Loss per state (one primitive — soft CE — for both terms):
    ce_target = -(target · log_softmax(logits))      # toward the improvement target
    ce_prior  = -(prior  · log_softmax(logits))      # anchor toward the clean prior
  --anchor-mode everywhere:  mean(weight·ce_target) + β·mean(ce_prior)
  --anchor-mode partition:   mean_corr(weight·ce_target) + β·mean_noncorr(ce_prior)
NO target sharpening (it was harmful in the strong-policy regime).

    PYTHONPATH=. python alphatrain/train_gumbel.py \\
        --tensor-file alphatrain/data/v15_pillar3f_slim.pt \\
        --resume alphatrain/data/pillar3f.pt --warm-start \\
        --epochs 12 --batch-size 16384 --lr 2e-4 --beta 1.0 \\
        --save-dir checkpoints/pillar3g2 \\
        --copy-to /content/drive/MyDrive/alphatrain/pillar3g2_best.pt
"""
import argparse
import os
import shutil
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alphatrain.dataset import GumbelDatasetGPU
from alphatrain.gumbel import NEG_INF
from alphatrain.model import AlphaTrainNet, count_parameters


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


def gumbel_losses(logits, target, prior, support, weight, beta, anchor_mode, ce_mode):
    """Return (loss, stats). logits/target/prior/support (B,6561); weight (B,).

    Reviewed loss (Gemini + ChatGPT 2026-06-16); default is simply
        L = mean(weight * CE_restricted(student, target)).
    The improvement CE is CANDIDATE-RESTRICTED (ce_mode='candidate'): the student
    log-softmax is taken over the per-state candidate support only, so the student is
    NOT forced to drive its ~7% off-top-10 mass to zero (audit_gumbel_target measured
    top-10 prior mass at only ~0.93 — full-softmax there would re-sharpen, the very
    failure we escaped). Because the target equals the prior EXACTLY on the ~96% non-
    correction states, mean(w*CE) is automatically ChatGPT's partition (CE-to-target on
    the ~4%, CE-to-prior anchor on the rest). `beta>0` adds an explicit FULL-softmax
    forward-CE-to-prior anchor (preserve broad behavior); never reverse KL.
    """
    logp_full = F.log_softmax(logits, dim=1)
    if ce_mode == 'candidate':
        logp_t = F.log_softmax(logits.masked_fill(support <= 0, NEG_INF), dim=1)
    else:
        logp_t = logp_full
    ce_target = -(target * logp_t).sum(1)        # (B,) candidate-restricted improvement
    ce_prior = -(prior * logp_full).sum(1)       # (B,) full-softmax forward anchor/monitor
    is_corr = weight > 1.0
    if anchor_mode == 'partition':
        nc = (~is_corr)
        ce_term = (weight * ce_target)[is_corr].sum() / max(int(is_corr.sum()), 1)
        anchor = ce_prior[nc].sum() / max(int(nc.sum()), 1)
        loss = ce_term + (beta if beta > 0 else 1.0) * anchor
    else:  # default: weighted CE-to-target everywhere (target==prior off-corrections)
        loss = (weight * ce_target).mean() + beta * ce_prior.mean()
    with torch.no_grad():
        stats = {
            'ce_target': float(ce_target.mean()),
            'ce_prior': float(ce_prior.mean()),
            'corr_frac': float(is_corr.float().mean()),
            'corr_match': 0.0,
        }
        if is_corr.any():
            # match rate on correction states: does the student's (restricted) argmax
            # agree with the improvement target's argmax? (the signal we want to move)
            ma = logp_t[is_corr].argmax(1) == target[is_corr].argmax(1)
            stats['corr_match'] = float(ma.float().mean())
    return loss, stats


def train_epoch(model, loader, optimizer, device, beta, anchor_mode, ce_mode,
                log_interval=100):
    model.train()
    tot, n, t0 = 0.0, 0, time.time()
    acc = {'ce_target': 0.0, 'ce_prior': 0.0, 'corr_frac': 0.0, 'corr_match': 0.0}
    nmatch = 0
    for bi, (obs, target, prior, support, weight) in enumerate(loader):
        obs = obs.to(device, non_blocking=True)
        out = model(obs)
        loss, st = gumbel_losses(_logits(out), target, prior, support, weight,
                                 beta, anchor_mode, ce_mode)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tot += loss.item(); n += 1
        for k in ('ce_target', 'ce_prior', 'corr_frac'):
            acc[k] += st[k]
        if st['corr_frac'] > 0:
            acc['corr_match'] += st['corr_match']; nmatch += 1
        if (bi + 1) % log_interval == 0:
            el = time.time() - t0
            print(f"    step {bi+1}/{len(loader)} loss={tot/n:.4f} "
                  f"ce_t={acc['ce_target']/n:.4f} ce_p={acc['ce_prior']/n:.4f} "
                  f"corr%={100*acc['corr_frac']/n:.2f} "
                  f"corr_match={acc['corr_match']/max(nmatch,1):.3f} "
                  f"[{el:.0f}s, {n*obs.shape[0]/el:.0f} st/s]", flush=True)
    return tot / max(n, 1), {k: acc[k] / max(n, 1) for k in acc}


@torch.no_grad()
def evaluate(model, loader, device, beta, anchor_mode, ce_mode):
    model.train(False)   # eval mode (avoid builtin-eval name; equivalent to model.eval())
    tot, n = 0.0, 0
    acc = {'ce_target': 0.0, 'ce_prior': 0.0, 'corr_match': 0.0}
    nmatch = 0
    for obs, target, prior, support, weight in loader:
        obs = obs.to(device, non_blocking=True)
        loss, st = gumbel_losses(_logits(model(obs)), target, prior, support, weight,
                                 beta, anchor_mode, ce_mode)
        tot += float(loss); n += 1
        acc['ce_target'] += st['ce_target']; acc['ce_prior'] += st['ce_prior']
        if st['corr_frac'] > 0:
            acc['corr_match'] += st['corr_match']; nmatch += 1
    return tot / max(n, 1), (acc['ce_target'] / max(n, 1),
                             acc['corr_match'] / max(nmatch, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor-file', required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=16384)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup-epochs', type=int, default=1)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--val-split', type=float, default=0.05)
    p.add_argument('--no-color-augment', action='store_true')
    p.add_argument('--no-dihedral-augment', action='store_true')
    p.add_argument('--augment-factor', type=int, default=8)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--warm-start', action='store_true',
                   help='Load weights only; reset optimizer/scheduler.')
    p.add_argument('--save-dir', default='checkpoints/gumbel')
    p.add_argument('--copy-to', type=str, default=None)
    # --- completed-Q target knobs (review-tunable; see gumbel.completed_q_target) ---
    p.add_argument('--visit-floor', type=float, default=20.0,
                   help='Visit floor for correction ELIGIBILITY only (not the target).')
    p.add_argument('--tau', type=float, default=0.02,
                   help='Advantage temperature. Must overcome the prior logit gap to '
                        'flip a correction (~0.01-0.025); set from audit_gumbel_target.')
    p.add_argument('--kappa', type=float, default=15.0,
                   help='Bayesian shrinkage strength of Q toward root_value (prior visits).')
    p.add_argument('--gamma', type=float, default=10.0)
    p.add_argument('--spread-gate', type=float, default=0.05)
    p.add_argument('--beta', type=float, default=0.0,
                   help='EXTRA explicit forward-CE-to-prior anchor weight. 0 = off '
                        '(the gated target already anchors the 96% to the prior).')
    p.add_argument('--anchor-mode', choices=['everywhere', 'partition'],
                   default='everywhere')
    p.add_argument('--ce-mode', choices=['candidate', 'full'], default='candidate',
                   help="Improvement CE support. 'candidate' (default) = restricted to "
                        "the searched candidates (top-10 prior mass is only ~0.93, so "
                        "full-softmax would re-sharpen the off-candidate tail).")
    p.add_argument('--max-train-states', type=int, default=0,
                   help='Cap base training states (local smoke only; 0 = all).')
    p.add_argument('--log-interval', type=int, default=100,
                   help='Steps between progress lines.')
    args = p.parse_args()

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f"Device: {device}", flush=True)

    gk = dict(visit_floor=args.visit_floor, tau=args.tau, gamma=args.gamma,
              spread_gate=args.spread_gate, kappa=args.kappa)
    train_set, val_set = GumbelDatasetGPU.make_train_val_split(
        args.tensor_file, val_split=args.val_split,
        augment=not args.no_dihedral_augment,
        color_augment=not args.no_color_augment,
        augment_factor=args.augment_factor, device=str(device), seed=42, **gk)
    if args.max_train_states and args.max_train_states < len(train_set.base_indices):
        train_set.base_indices = train_set.base_indices[:args.max_train_states]
        print(f"  SUBSAMPLE: train base capped to {len(train_set.base_indices):,}",
              flush=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=train_set.collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=0, collate_fn=val_set.collate)
    print(f"Train={len(train_set):,} Val={len(val_set):,} | target knobs {gk} "
          f"beta={args.beta} anchor={args.anchor_mode}", flush=True)

    model = AlphaTrainNet(num_blocks=args.num_blocks, channels=args.channels).to(device)
    print(f"Model: {args.num_blocks}b x {args.channels}ch, "
          f"{count_parameters(model):,} params", flush=True)

    start_epoch, best_val, ckpt = 0, float('inf'), None
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt['model']
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        ms = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in ms and v.shape == ms[k].shape}
        model.load_state_dict(filtered, strict=False)
        skipped = [k for k in state if k not in filtered]
        if skipped:
            print(f"  Skipped {len(skipped)} keys (shape/name mismatch)", flush=True)
        if not args.warm_start and 'optimizer' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            best_val = ckpt.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch - 1}", flush=True)
        else:
            print(f"Warm-start from epoch {ckpt.get('epoch', '?')}, fresh optimizer",
                  flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs])
    if args.resume and not args.warm_start and ckpt is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\n=== Training {args.epochs} epochs (start {start_epoch}) ===", flush=True)
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.2e})", flush=True)
        et = time.time()
        tl, tst = train_epoch(model, train_loader, optimizer, device,
                              args.beta, args.anchor_mode, args.ce_mode,
                              log_interval=args.log_interval)
        vl, (v_ce_t, v_match) = evaluate(model, val_loader, device,
                                         args.beta, args.anchor_mode, args.ce_mode)
        scheduler.step()
        print(f"  Train: loss={tl:.4f} ce_t={tst['ce_target']:.4f} "
              f"corr_match={tst['corr_match']:.3f} | "
              f"Val: loss={vl:.4f} ce_t={v_ce_t:.4f} corr_match={v_match:.3f} "
              f"[{time.time()-et:.0f}s]", flush=True)

        ck = {'epoch': epoch, 'model': model.state_dict(),
              'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
              'val_loss': vl, 'best_val_loss': min(best_val, vl),
              'args': vars(args)}
        latest = os.path.join(args.save_dir, 'latest.pt')
        epoch_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pt')
        torch.save(ck, latest); torch.save(ck, epoch_path)
        if args.copy_to:
            run_prefix = os.path.basename(args.save_dir.rstrip('/'))
            cd = os.path.dirname(args.copy_to) or '.'
            shutil.copy2(epoch_path, os.path.join(cd, f'{run_prefix}_epoch_{epoch+1}.pt'))
        if vl < best_val:
            best_val = vl
            torch.save(ck, os.path.join(args.save_dir, 'best.pt'))
            print(f"  ** New best (val={vl:.4f}) **", flush=True)
            if args.copy_to:
                shutil.copy2(os.path.join(args.save_dir, 'best.pt'), args.copy_to)
    print(f"\n=== Done. Best val: {best_val:.4f} ===", flush=True)


if __name__ == '__main__':
    main()
