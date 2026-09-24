"""Same-state conservative policy improvement for frontier corrections.

Hard CE repeatedly drove the 128-channel policy toward near-one-hot outputs,
including on rows where search already agreed with the base.  This trainer
instead uses, on every state, the frozen base distribution as the anchor and
adds a bounded one-hot improvement term only on prevalidated disagreements:

    L(s) = KL(p_base(.|s) || p_student(.|s))
           + eta(s) * -log p_student(a_search|s)

For an unconstrained categorical policy the exact optimum on an edited state
is ``(p_base + eta * one_hot(a_search)) / (1 + eta)``.  Thus eta is a direct
mass/KL budget, not an optimizer-dependent interpolation discovered after a
hard-CE endpoint.  Agreement and broad rows receive only zero-at-initialization
preservation loss.

The first intended use is r2_frontier with edits restricted to successful
crisis rows, full-legal base disagreements, and high search visit share.  BN is
frozen to the base's inference statistics for the entire run.
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit
from alphatrain.train_path_b import frozen_bn


def conservative_policy_loss(base_logits, student_logits, teacher_move,
                             edit_weight):
    """Return scalar loss plus detached per-row KL and teacher NLL.

    ``edit_weight`` is eta per row (zero for preservation-only rows).
    The base branch is detached by construction even if the caller forgot to
    wrap it in ``no_grad``.
    """
    base_logp = F.log_softmax(base_logits.detach().float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    base_p = base_logp.exp()
    kl = (base_p * (base_logp - student_logp)).sum(dim=-1)
    teacher_nll = -student_logp.gather(
        1, teacher_move.long().unsqueeze(1)).squeeze(1)
    per_row = kl + edit_weight.float() * teacher_nll
    return per_row.mean(), kl.detach(), teacher_nll.detach()


def make_frozen_base_and_student(model):
    """Return an eval-only base and an independently trainable exact clone."""
    model.train(False)
    model.requires_grad_(False)
    student = copy.deepcopy(model)
    student.requires_grad_(True)
    student.train(True)
    return model, student


@torch.inference_mode()
def functional_audit(model, reference_model, ds, indices, teacher_move,
                     device, batch_size):
    """Same-protocol legal argmax adoption/drift on fixed rows.

    The reference argmax is recomputed in the same dtype and batch as the
    student.  Comparing with a sidecar generated in fp16 or another batch
    shape would incorrectly count numerical protocol changes as training
    drift.
    """
    if not len(indices):
        return {'n': 0, 'teacher_adopt': float('nan'),
                'base_retain': float('nan'), 'mean_kl': float('nan'),
                'p90_kl': float('nan')}
    dtype = next(model.parameters()).dtype
    adopted = retained = seen = 0
    kls = []
    for start in range(0, len(indices), batch_size):
        ix_np = indices[start:start + batch_size]
        ix = torch.from_numpy(ix_np).to(device)
        obs = ds._build_obs_core(
            ds.boards[ix], next_pos=ds.next_pos[ix],
            next_col=ds.next_col[ix], n_next=ds.n_next[ix])
        # Audits measure deployed behavior and must neither use batch
        # statistics nor mutate running buffers when the caller is between
        # training epochs.
        with frozen_bn(model):
            out = model(obs.to(dtype))
        logits_t = (out[0] if isinstance(out, tuple) else out).float()
        ref_out = reference_model(obs.to(dtype))
        ref_logits = (ref_out[0] if isinstance(ref_out, tuple)
                      else ref_out).float()
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        logp = F.log_softmax(logits_t, dim=-1)
        kls.extend((ref_logp.exp() * (ref_logp - logp)).sum(-1)
                   .cpu().tolist())
        logits = logits_t.cpu().numpy()
        ref_logits_np = ref_logits.cpu().numpy()
        boards = ds.boards[ix].cpu().numpy().astype(np.int8)
        for j, row in enumerate(ix_np):
            k, actions, _ = _legal_priors_jit(boards[j], logits[j], 1)
            ref_k, ref_actions, _ = _legal_priors_jit(
                boards[j], ref_logits_np[j], 1)
            if not k or not ref_k:
                continue
            argmax = int(actions[0])
            adopted += argmax == int(teacher_move[row])
            retained += argmax == int(ref_actions[0])
            seen += 1
    return {'n': seen, 'teacher_adopt': adopted / max(seen, 1),
            'base_retain': retained / max(seen, 1),
            'mean_kl': float(np.mean(kls)),
            'p90_kl': float(np.percentile(kls, 90))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', required=True)
    p.add_argument('--strata', required=True)
    p.add_argument('--base-policy-sidecar', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--eta', type=float, default=0.2)
    p.add_argument('--min-visit-share', type=float, default=0.36)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=20260808)
    p.add_argument('--device', default=None)
    p.add_argument('--max-states', type=int, default=0,
                   help='Random corpus cap for smoke tests; 0 uses all rows.')
    p.add_argument('--dry-run', action='store_true',
                   help='Validate inputs and print edit counts without training.')
    p.add_argument('--audit-rows', type=int, default=5000)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--save-dir', default='checkpoints/conservative_frontier')
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f'device={device}', flush=True)

    side = np.load(args.base_policy_sidecar, allow_pickle=False)
    strata = np.load(args.strata, allow_pickle=False)['strata'].astype(str)
    base_move = side['base_argmax'].astype(np.int64)
    teacher_move = side['target_argmax'].astype(np.int64)
    visit_share = side['target_top_share'].astype(np.float32)
    disagree = side['disagree'].astype(bool)
    n = len(base_move)
    if not (len(strata) == len(teacher_move) == len(visit_share) == n):
        raise ValueError('tensor/strata/base-policy sidecar row mismatch')
    crisis = np.char.startswith(strata, 'f_')
    broad = np.char.startswith(strata, 'b_')
    valid = (base_move >= 0) & (teacher_move >= 0)
    edit = (crisis & valid & disagree
            & (visit_share >= args.min_visit_share))
    print(f'rows={n:,}; crisis={crisis.sum():,}; broad={broad.sum():,}',
          flush=True)
    print(f'disagreements: crisis={(crisis & valid & disagree).sum():,}; '
          f'broad={(broad & valid & disagree).sum():,}', flush=True)
    print(f'edits={edit.sum():,} ({100*edit.mean():.2f}%) '
          f'eta={args.eta:g} mass-dose={args.eta/(1+args.eta):.1%}',
          flush=True)
    protocol = str(side['protocol']) if 'protocol' in side else 'unknown'
    print(f'base-policy sidecar: {protocol}', flush=True)
    if protocol != 'unknown' and args.base not in protocol:
        raise ValueError(f'sidecar was not built from requested base {args.base}')
    if args.dry_run:
        return

    ds = TensorDatasetGPU(args.tensor, augment=False, color_augment=False,
                          augment_factor=1, device=str(device))
    if ds.boards.shape[0] != n:
        raise ValueError(f'tensor rows {ds.boards.shape[0]:,} != sidecar {n:,}')

    loaded, _ = load_model(args.base, device, fp16=False)
    base, student = make_frozen_base_and_student(loaded)
    base_state = {k: v.detach().clone()
                  for k, v in student.state_dict().items()}

    edit_t = torch.from_numpy(edit).to(device)
    teacher_t = torch.from_numpy(teacher_move).to(device)
    all_rows = np.arange(n, dtype=np.int64)
    if args.max_states and args.max_states < n:
        all_rows = np.sort(rng.choice(
            n, size=args.max_states, replace=False)).astype(np.int64)
        if not edit[all_rows].any():
            raise ValueError('--max-states sample contains no edit rows')
        print(f'smoke subset={len(all_rows):,}; edits={edit[all_rows].sum():,}',
              flush=True)

    edit_rows = np.flatnonzero(edit)
    broad_rows = np.flatnonzero(broad)
    audit_edit = np.sort(rng.choice(
        edit_rows, size=min(args.audit_rows, len(edit_rows)), replace=False))
    audit_broad = np.sort(rng.choice(
        broad_rows, size=min(args.audit_rows, len(broad_rows)), replace=False))

    # Quantify any disagreement between the sidecar's generation protocol
    # (often fp16) and this trainer's reference forward before attributing
    # teacher adoption to learning.
    base_edit = functional_audit(student, base, ds, audit_edit, teacher_move,
                                 device, args.batch_size)
    base_broad = functional_audit(student, base, ds, audit_broad, teacher_move,
                                  device, args.batch_size)
    print(f'[preflight] edit teacher-agree={base_edit["teacher_adopt"]:.3f}; '
          f'student/base retain={base_edit["base_retain"]:.3f}; '
          f'broad retain={base_broad["base_retain"]:.3f}; '
          f'KL={base_broad["mean_kl"]:.6f}', flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    os.makedirs(args.save_dir, exist_ok=True)
    steps = (len(all_rows) + args.batch_size - 1) // args.batch_size

    for epoch in range(1, args.epochs + 1):
        order = all_rows[rng.permutation(len(all_rows))]
        total_loss = total_kl = total_edit_nll = 0.0
        total_rows = total_edits = 0
        t0 = time.time()
        for step in range(steps):
            ix_np = order[step * args.batch_size:(step + 1) * args.batch_size]
            ix = torch.from_numpy(ix_np).to(device)
            obs = ds._build_obs_core(
                ds.boards[ix], next_pos=ds.next_pos[ix],
                next_col=ds.next_col[ix], n_next=ds.n_next[ix])
            with torch.inference_mode():
                b_out = base(obs)
                base_logits = b_out[0] if isinstance(b_out, tuple) else b_out
            with frozen_bn(student):
                s_out = student(obs)
                student_logits = s_out[0] if isinstance(s_out, tuple) else s_out
            weights = edit_t[ix].float() * args.eta
            loss, kl, nll = conservative_policy_loss(
                base_logits, student_logits, teacher_t[ix], weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            n_batch = len(ix_np)
            n_edit = int(edit[ix_np].sum())
            total_loss += float(loss.detach()) * n_batch
            total_kl += float(kl.sum())
            if n_edit:
                total_edit_nll += float(nll[edit_t[ix]].sum())
            total_rows += n_batch
            total_edits += n_edit
            if (step + 1) % args.log_every == 0 or step + 1 == steps:
                elapsed = time.time() - t0
                eta_s = elapsed / (step + 1) * (steps - step - 1)
                print(f'  ep{epoch} {step+1}/{steps} '
                      f'loss={total_loss/total_rows:.5f} '
                      f'KL={total_kl/total_rows:.5f} '
                      f'editNLL={total_edit_nll/max(total_edits,1):.3f} '
                      f'edits={total_edits:,} ETA={eta_s:.0f}s', flush=True)

        student.train(False)
        ea = functional_audit(student, base, ds, audit_edit, teacher_move,
                              device, args.batch_size)
        ba = functional_audit(student, base, ds, audit_broad, teacher_move,
                              device, args.batch_size)
        print(f'[epoch {epoch}] edit adoption={ea["teacher_adopt"]:.3f} '
              f'(base-retain={ea["base_retain"]:.3f}, '
              f'KL={ea["mean_kl"]:.5f}/{ea["p90_kl"]:.5f} mean/P90, '
              f'n={ea["n"]:,}); '
              f'broad argmax-retain={ba["base_retain"]:.3f} '
              f'KL={ba["mean_kl"]:.5f}/{ba["p90_kl"]:.5f} mean/P90 '
              f'(n={ba["n"]:,})', flush=True)

        sd = student.state_dict()
        for k in sd:
            if k.endswith('.running_mean') or k.endswith('.running_var'):
                if not torch.equal(sd[k], base_state[k]):
                    raise AssertionError(f'BN buffer drifted: {k}')
        out = os.path.join(args.save_dir, f'epoch_{epoch}.pt')
        torch.save({'model': sd, 'epoch': epoch, 'args': vars(args),
                    'policy_only': True,
                    'selection': {'n_rows': n, 'n_edits': int(edit.sum()),
                                  'protocol': protocol}}, out)
        print(f'saved {out} (BN buffers == base)', flush=True)
        student.train(True)


if __name__ == '__main__':
    main()
