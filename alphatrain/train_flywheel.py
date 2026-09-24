"""Lineage-pure play/mine/train policy iteration on all recorded states.

Two objectives allow a controlled retry of old mixtures on the stronger vh3
base and corrected corpus semantics:

``bounded`` (primary)
    Every row preserves the frozen base distribution.  Search rows add a
    bounded soft/hard visit target:

        KL(base || student) + eta * CE(search_mix, student)

    The exact per-state probability-space target is
    ``(base + eta*search_mix)/(1+eta)``.

``legacy`` (control)
    Search rows receive ordinary soft/hard CE; greedy anchor rows receive hard
    CE on the recorded base action.  With all rows consumed once, the current
    vh3 artifacts naturally reproduce the old ~6:1 rehearsal mixture without
    importing big-model games.

``aggregate`` (plastic policy iteration)
    Search-trajectory rows receive ordinary soft/hard teacher CE. Replay
    anchors preserve the frozen base's complete distribution with KL rather
    than sharpening its recorded argmax. This objective may warm-start or
    reset the same architecture from scratch and may update BatchNorm.

BatchNorm running statistics are frozen by default. Aggregate and legacy arms
have an explicit train-mode opt-in; bounded updates keep deployment statistics
fixed. Train/validation splits come from source-game groups stored by
``build_flywheel_corpus``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model
from alphatrain.scripts.fleet_gpu import gpu_legal_mask


def _atomic_torch_save(payload, path):
    """Write a checkpoint atomically so an interrupted run stays resumable."""
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _atomic_json_save(payload, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(tmp, path)


def _sha256(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _deployment_numerics(model, dtype):
    """Summarize state which becomes non-finite at deployment precision.

    MPS inference converts the entire module, including BatchNorm buffers, to
    fp16.  Autocast training leaves those buffers in fp32, so a running
    variance above 65,504 can look healthy in the training process and become
    ``inf`` in the actual actor.  Keep this diagnostic beside every epoch.
    """
    cast_nonfinite = source_nonfinite = 0
    bn_var_cast_nonfinite = bn_var_source_nonfinite = 0
    bn_var_max_finite = 0.0
    for name, value in model.state_dict().items():
        if not torch.is_floating_point(value):
            continue
        source_nonfinite += int((~torch.isfinite(value)).sum().item())
        cast = value.to(dtype=dtype)
        cast_nonfinite += int((~torch.isfinite(cast)).sum().item())
        if name.endswith('running_var'):
            finite = value[torch.isfinite(value)]
            if finite.numel():
                bn_var_max_finite = max(
                    bn_var_max_finite, float(finite.max().item()))
            bn_var_source_nonfinite += int(
                (~torch.isfinite(value)).sum().item())
            bn_var_cast_nonfinite += int(
                (~torch.isfinite(cast)).sum().item())
    return {
        'dtype': str(dtype),
        'source_nonfinite': source_nonfinite,
        'cast_nonfinite': cast_nonfinite,
        'bn_running_var_source_nonfinite': bn_var_source_nonfinite,
        'bn_running_var_cast_nonfinite': bn_var_cast_nonfinite,
        'bn_running_var_max_finite': bn_var_max_finite,
    }


def _deployment_copy(model, dtype):
    """Return an eval-only copy with weights and BN buffers truly cast."""
    deployed = copy.deepcopy(model).train(False)
    deployed.requires_grad_(False)
    if dtype != torch.float32:
        deployed.to(dtype=dtype)
    return deployed


def _quantize_frozen_bn_for_deployment_(model, dtype):
    """Put frozen BN buffers on the deployment grid without changing fp16.

    This may turn an fp32 variance above the fp16 range into ``inf``.  That is
    already what native deployment does, and keeping the fp32 master at the
    same value removes a large hidden train/deploy mismatch.  Never do this to
    train-mode BN: an infinite exponential-moving-average buffer cannot
    recover.
    """
    if dtype == torch.float32:
        return
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue
            for name in ('running_mean', 'running_var'):
                value = getattr(module, name, None)
                if value is not None:
                    value.copy_(value.to(dtype=dtype).to(dtype=value.dtype))


def _functional_gate(metrics, deployment_numerics, base_numerics, *,
                     min_retain=0.0, max_legal_kl=-1.0,
                     max_new_bn_nonfinite=-1):
    """Return observed deployment safety values and any gate failures."""
    retain = min(value['retain'] for value in metrics.values())
    legal_kl = max(value['mean_legal_kl'] for value in metrics.values())
    new_bn_nonfinite = max(
        0,
        deployment_numerics['bn_running_var_cast_nonfinite']
        - base_numerics['bn_running_var_cast_nonfinite'])
    failures = []
    if min_retain and (not math.isfinite(retain)
                       or retain < min_retain):
        failures.append(
            f'retain={retain:.6g} < {min_retain:.6g}')
    if max_legal_kl >= 0 and (not math.isfinite(legal_kl)
                              or legal_kl > max_legal_kl):
        failures.append(
            f'legal_kl={legal_kl:.6g} > {max_legal_kl:.6g}')
    if (max_new_bn_nonfinite >= 0
            and new_bn_nonfinite > max_new_bn_nonfinite):
        failures.append(
            f'new_bn_nonfinite={new_bn_nonfinite} > '
            f'{max_new_bn_nonfinite}')
    return {
        'min_retain': retain,
        'max_mean_legal_kl': legal_kl,
        'new_bn_running_var_cast_nonfinite': new_bn_nonfinite,
        'thresholds': {
            'min_retain': min_retain,
            'max_legal_kl': max_legal_kl,
            'max_new_bn_nonfinite': max_new_bn_nonfinite,
        },
        'failures': failures,
        'passed': not failures,
    }


def _capture_rng_state(loader_rng, numpy_rng, device):
    state = {
        'torch_cpu': torch.get_rng_state(),
        'loader': loader_rng.get_state(),
        'numpy': numpy_rng.bit_generator.state,
    }
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state_all()
    elif device.type == 'mps' and hasattr(torch.mps, 'get_rng_state'):
        state['mps'] = torch.mps.get_rng_state()
    return state


def _restore_rng_state(state, loader_rng, numpy_rng, device):
    """Restore sampling/augmentation streams saved at an epoch boundary."""
    if not state:
        return False
    # Loading a resumable checkpoint with map_location=mps/cuda also moves
    # these byte tensors, but RNG setter APIs require CPU ByteTensors.
    torch.set_rng_state(state['torch_cpu'].cpu())
    loader_rng.set_state(state['loader'].cpu())
    numpy_rng.bit_generator.state = state['numpy']
    if device.type == 'cuda' and 'cuda' in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    elif (device.type == 'mps' and 'mps' in state
          and hasattr(torch.mps, 'set_rng_state')):
        torch.mps.set_rng_state(state['mps'].cpu())
    return True


def _completed_epoch(checkpoint, path):
    """Read new metadata, with a filename fallback for early checkpoints."""
    if 'completed_epoch' in checkpoint:
        return int(checkpoint['completed_epoch'])
    match = re.search(r'(?:^|/)epoch_(\d+)\.pt$', path)
    if match:
        return int(match.group(1))
    raise ValueError('resume checkpoint has no completed_epoch; resume from '
                     'an epoch_N.pt or a new latest.pt checkpoint')


def _validate_resume_args(saved, current):
    """Reject a resume that would silently change the optimization problem."""
    critical = (
        'targets', 'anchors', 'base', 'objective', 'eta', 'soft_alpha',
        'base_policy_sidecar', 'sidecar_disagree_key',
        'edit_weight_sidecar', 'edit_weight_key',
        'source_weights', 'base_agree_weight', 'base_disagree_weight',
        'edit_exposure',
        'update_bn', 'anchor_weight', 'from_scratch', 'epochs', 'batch_size',
        'lr', 'weight_decay', 'schedule', 'warmup_fraction', 'min_lr_ratio',
        'seed', 'precision', 'legal_support_loss', 'no_dihedral_augment',
        'color_augment', 'augment_factor', 'max_target_states',
        'max_anchor_states', 'min_target_edits',
    )
    mismatches = []
    for key in critical:
        if key in saved and saved[key] != current.get(key):
            mismatches.append(
                f'{key}: saved={saved[key]!r}, current={current.get(key)!r}')
    if mismatches:
        raise ValueError('resume configuration mismatch:\n  '
                         + '\n  '.join(mismatches))


def flywheel_loss(base_logits, student_logits, policy_target, hard_target_move,
                  target_weight, *, objective, eta, soft_alpha, is_target,
                  anchor_weight=1.0, legal_mask=None, row_weight=None):
    """Return loss and detached per-row diagnostics for one source batch."""
    student_float = student_logits.float()
    if legal_mask is not None:
        legal_mask = legal_mask.to(student_float.device).bool()
        student_float = student_float.masked_fill(
            ~legal_mask, float('-inf'))
    student_logp = F.log_softmax(student_float, dim=-1)
    if objective in ('bounded', 'aggregate'):
        base_float = base_logits.detach().float()
        if legal_mask is not None:
            base_float = base_float.masked_fill(
                ~legal_mask, float('-inf'))
        base_logp = F.log_softmax(base_float, dim=-1)
        base_p = base_logp.exp()
        kl_terms = base_p * (base_logp - student_logp)
        if legal_mask is not None:
            # Outside the support this is 0 * (inf-inf), which is NaN unless
            # explicitly removed.
            kl_terms = torch.where(
                legal_mask, kl_terms, torch.zeros_like(kl_terms))
        kl = kl_terms.sum(-1)
        per = kl
    else:
        kl = torch.zeros(student_logits.shape[0], device=student_logits.device)
        per = torch.zeros_like(kl)

    soft_terms = policy_target.float() * student_logp
    if legal_mask is not None:
        soft_terms = torch.where(
            legal_mask, soft_terms, torch.zeros_like(soft_terms))
    soft_nll = -soft_terms.sum(-1)
    hard_nll = -student_logp.gather(
        1, hard_target_move.long().unsqueeze(1)).squeeze(1)
    mixed_nll = soft_alpha * soft_nll + (1.0 - soft_alpha) * hard_nll

    if torch.is_tensor(is_target):
        target_mask = is_target.to(student_logits.device).bool()
    else:
        target_mask = torch.full(
            (student_logits.shape[0],), bool(is_target),
            dtype=torch.bool, device=student_logits.device)
    if objective == 'bounded':
        per = per + (eta * target_weight.float() * mixed_nll
                     * target_mask.float())
    elif objective == 'aggregate':
        # A plastic policy-iteration step: imitate rolling search without a
        # per-target trust region, while preserving the base distribution on
        # broad replay anchors. This avoids legacy one-hot sharpening of the
        # champion's recorded argmax.
        per = torch.where(
            target_mask, target_weight.float() * mixed_nll,
            float(anchor_weight) * kl)
    else:
        # Historical rehearsal semantics in a genuinely mixed batch: search
        # rows imitate their teacher while anchors imitate the current policy's
        # recorded greedy action as a one-hot target.
        per = torch.where(
            target_mask, target_weight.float() * mixed_nll,
            float(anchor_weight) * hard_nll)
    if row_weight is None:
        loss = per.mean()
    else:
        row_weight = row_weight.to(per.device).float()
        loss = (per * row_weight).sum() / row_weight.sum().clamp_min(1e-12)
    return loss, kl.detach(), mixed_nll.detach()


def legal_mask_from_observation(obs):
    """Reconstruct integer boards and return the deployment legal support."""
    occupied = obs[:, :7].sum(1) > 0.5
    colors = obs[:, :7].argmax(1).to(torch.int8) + 1
    boards = torch.where(occupied, colors, torch.zeros_like(colors))
    return gpu_legal_mask(boards)


def make_datasets(path, device, *, augment, color_augment, augment_factor,
                  train):
    probe = TensorDatasetGPU(path, augment=augment if train else False,
                             color_augment=color_augment if train else False,
                             augment_factor=(augment_factor if train else 1),
                             device=str(device))
    if probe.split is None:
        raise ValueError(f'{path} has no game-group split')
    wanted = 0 if train else 1
    probe.base_indices = (probe.split == wanted).nonzero(as_tuple=True)[0]
    probe.return_flywheel = True
    return probe


def cap_dataset(ds, max_states, rng, min_edits=0):
    """Randomly cap states, optionally reserving declared edit coverage."""
    if not max_states or max_states >= len(ds.base_indices):
        return
    n = len(ds.base_indices)
    if min_edits:
        if ds.flywheel_disagree is None:
            raise ValueError('minimum edit coverage needs an edit sidecar')
        edit_positions = (ds.flywheel_disagree[ds.base_indices]
                          .nonzero(as_tuple=True)[0].cpu().numpy())
        n_edit = min(int(min_edits), len(edit_positions), max_states)
        reserved = (edit_positions if n_edit == len(edit_positions)
                    else rng.choice(edit_positions, n_edit, replace=False))
        used = np.zeros(n, dtype=bool)
        used[reserved] = True
        pool = np.flatnonzero(~used)
        fill = rng.choice(
            pool, size=max_states - len(reserved), replace=False)
        positions = np.sort(np.concatenate((reserved, fill))).astype(np.int64)
        print(f'  capped dataset to {max_states:,} rows with '
              f'{len(reserved):,}/{len(edit_positions):,} declared edits',
              flush=True)
    else:
        positions = np.sort(rng.choice(
            n, size=max_states, replace=False)).astype(np.int64)
    positions = torch.from_numpy(positions).to(ds.device)
    ds.base_indices = ds.base_indices[positions]


class TensorBatchLoader:
    """Batch GPU-backed datasets without materializing Python integer lists.

    PyTorch's default random sampler converts a full ``randperm`` to a Python
    list.  The 19.6M-row anchor corpus makes that needlessly expensive.  This
    iterator keeps the permutation as one CPU tensor and hands tensor slices
    directly to ``TensorDatasetGPU.collate``.
    """

    def __init__(self, dataset, batch_size, *, shuffle, generator=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        order = (torch.randperm(n, generator=self.generator)
                 if self.shuffle else None)
        for start in range(0, n, self.batch_size):
            if order is None:
                indices = torch.arange(
                    start, min(start + self.batch_size, n), dtype=torch.long)
            else:
                indices = order[start:start + self.batch_size]
            yield self.dataset.collate(indices)


class PooledBatchLoader:
    """Mix target and anchor rows *within* every shuffled batch.

    Alternating source-pure batches has the same first-moment objective under
    plain SGD, but very different Adam moment dynamics: one correction update
    is followed by several preservation-only pullbacks.  A random permutation
    of the pooled row space implements the declared all-data mixture directly
    and still consumes every target and anchor exactly once per epoch.
    """

    def __init__(self, target, anchor, batch_size, *, shuffle, generator=None):
        self.datasets = (target, anchor)
        self.sizes = (len(target), len(anchor))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self):
        return math.ceil(sum(self.sizes) / self.batch_size)

    def __iter__(self):
        return self.iter_from()

    def iter_from(self, start_batch=0, epoch_seed=None):
        """Iterate a reproducible epoch, optionally after a saved batch.

        A per-epoch seed makes the row order reconstructible without storing a
        multi-million-element permutation in every resumable checkpoint.
        Skipped batches do not call dataset.collate, so restored augmentation
        RNG state continues exactly at the next unfinished batch.
        """
        n_target, n_anchor = self.sizes
        n = n_target + n_anchor
        generator = self.generator
        if epoch_seed is not None:
            generator = torch.Generator().manual_seed(int(epoch_seed))
        order = (torch.randperm(n, generator=generator)
                 if self.shuffle else torch.arange(n))
        for batch_i, start in enumerate(range(0, n, self.batch_size)):
            if batch_i < start_batch:
                continue
            pooled = order[start:start + self.batch_size]
            target_pos = pooled[pooled < n_target]
            anchor_pos = pooled[pooled >= n_target] - n_target
            parts = []
            masks = []
            if len(target_pos):
                batch = self.datasets[0].collate(target_pos)
                parts.append(batch)
                masks.append(torch.ones(
                    len(target_pos), dtype=torch.bool,
                    device=batch[0].device))
            if len(anchor_pos):
                batch = self.datasets[1].collate(anchor_pos)
                parts.append(batch)
                masks.append(torch.zeros(
                    len(anchor_pos), dtype=torch.bool,
                    device=batch[0].device))
            combined = tuple(torch.cat(values, dim=0)
                             for values in zip(*parts))
            yield (*combined, torch.cat(masks))


@torch.inference_mode()
def audit(student, base, loaders, device, max_batches=20,
          amp_dtype=torch.float32, bounded_eta=None,
          bounded_source_weights=None, bounded_agree_weight=1.0,
          bounded_disagree_weight=1.0, bounded_soft_alpha=0.0):
    was_training = student.training
    student.train(False)

    def module_dtype(module):
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            try:
                return next(module.buffers()).dtype
            except StopIteration:
                return torch.float32

    student_dtype = module_dtype(student)
    base_dtype = module_dtype(base)
    out = {}
    for name, loader in loaders.items():
        kls, legal_kls, retains, n = [], [], 0, 0
        teacher_before = teacher_after = edit_adopt = edit_n = 0
        teacher_n = 0
        agree_preserve = agree_n = 0
        bounded_opt_adopt = bounded_opt_edit_n = 0
        edit_probability_gaps = []
        bounded_residual_kls = []
        bounded_target_base_kls = []
        bounded_all_residual_kls = []
        bounded_all_target_base_kls = []
        bounded_all_base_target_kls = []
        bounded_all_action_changes = bounded_all_teacher_matches = 0
        bounded_all_n = 0
        edit_teacher_prob_before = edit_teacher_prob_after = 0.0
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            (obs, policy, target_weight, _, hard_target,
             source_id) = batch[:6]
            declared_edit = (batch[6] if len(batch) >= 7 else None)
            obs = obs.to(device, non_blocking=True)
            policy = policy.to(device, non_blocking=True)
            hard_target = hard_target.to(device).long()
            # Standalone audits load genuinely half-precision checkpoints and
            # must reproduce deployment with half inputs.  Wrapping an already
            # half model in autocast is not numerically identical on MPS (most
            # visibly around BatchNorm) and can flip near-tied legal argmaxes.
            # During training the models remain FP32, so autocast is still the
            # intended mixed-precision forward path there.
            if student_dtype != torch.float32 or base_dtype != torch.float32:
                s = student(obs.to(student_dtype))
                b = base(obs.to(base_dtype))
            else:
                use_amp = amp_dtype != torch.float32
                with torch.amp.autocast(device.type, dtype=amp_dtype,
                                        enabled=use_amp):
                    s = student(obs)
                    b = base(obs)
            sl = (s[0] if isinstance(s, tuple) else s).float()
            bl = (b[0] if isinstance(b, tuple) else b).float()
            blog = F.log_softmax(bl, -1)
            slog = F.log_softmax(sl, -1)
            kls.append((blog.exp() * (blog - slog)).sum(-1).cpu())
            # Deployment chooses from the exact legal action set.  Rebuild
            # the integer board from the one-hot observation so this audit
            # does not silently select an illegal full-softmax argmax.
            legal = legal_mask_from_observation(obs)
            # The recorded C++ engine is authoritative on the tiny winding-
            # corridor tail where the batched component labeller disagrees.
            # Match the training support: retain every recorded search action
            # and the declared hard target instead of manufacturing inf KL.
            legal = legal | (policy > 0)
            legal.scatter_(1, hard_target.unsqueeze(1), True)
            sl_legal = sl.masked_fill(~legal, float('-inf'))
            bl_legal = bl.masked_fill(~legal, float('-inf'))
            sl_legal_logp = F.log_softmax(sl_legal, -1)
            bl_legal_logp = F.log_softmax(bl_legal, -1)
            legal_kl = (bl_legal_logp.exp()
                        * (bl_legal_logp - sl_legal_logp))
            # Avoid 0 * (inf-inf) NaNs outside the legal support.
            legal_kl = torch.where(
                legal, legal_kl, torch.zeros_like(legal_kl)).sum(-1)
            legal_kls.append(legal_kl.cpu())
            sa, ba = sl_legal.argmax(1), bl_legal.argmax(1)
            # Temperature-sampled / explicitly disabled target rows are
            # preservation states, not teacher edits.  They still contribute
            # to KL and overall retention, but not adoption statistics.
            target_weight_device = target_weight.to(device)
            measured = ((target_weight_device > 0)
                        if name.startswith('target')
                        else torch.ones_like(hard_target, dtype=torch.bool))
            retains += int((sa == ba).sum())
            teacher_before += int(((ba == hard_target) & measured).sum())
            teacher_after += int(((sa == hard_target) & measured).sum())
            teacher_n += int(measured.sum())
            # When training attaches an immutable sidecar, audit the declared
            # optimization stratum rather than silently expanding it back to
            # every online base/teacher disagreement.  This matters for mined
            # subsets such as crisis-only edits.  The online comparison remains
            # the fallback for legacy tensors without a sidecar.
            if declared_edit is not None:
                edit = declared_edit.to(device).bool() & measured
                agree = ~declared_edit.to(device).bool() & measured
            else:
                edit = (ba != hard_target) & measured
                agree = (ba == hard_target) & measured
            edit_adopt += int(((sa == hard_target) & edit).sum())
            edit_n += int(edit.sum())
            agree_preserve += int(((sa == ba) & agree).sum())
            agree_n += int(agree.sum())
            if bounded_eta is not None and name.startswith('target'):
                base_p_legal = bl_legal_logp.exp()
                base_best_p = base_p_legal.gather(
                    1, ba.unsqueeze(1)).squeeze(1)
                teacher_p = base_p_legal.gather(
                    1, hard_target.unsqueeze(1)).squeeze(1)
                probability_gap = (base_best_p - teacher_p).clamp_min(0)
                if edit.any():
                    edit_probability_gaps.append(
                        probability_gap[edit].cpu())
                effective_eta = (target_weight_device.float()
                                 * float(bounded_eta))
                if bounded_source_weights is not None:
                    source_weight_tensor = torch.as_tensor(
                        bounded_source_weights, device=device,
                        dtype=torch.float32)
                    effective_eta *= source_weight_tensor[
                        source_id.to(device).long()]
                effective_eta *= torch.where(
                    edit, float(bounded_disagree_weight),
                    float(bounded_agree_weight))

                # The exact per-state optimum of
                #   KL(base || student) + eta * CE(search, student)
                # is the normalized mixture below.  Keep the actual target
                # mass in the denominator: sparse visit tensors are nearly,
                # but not bit-exactly, normalized after fp16 storage.
                search_target = policy.float() * float(bounded_soft_alpha)
                search_target.scatter_add_(
                    1, hard_target.unsqueeze(1),
                    torch.full(
                        (len(hard_target), 1),
                        1.0 - float(bounded_soft_alpha),
                        device=device, dtype=torch.float32))
                search_target = torch.where(
                    legal, search_target, torch.zeros_like(search_target))
                target_mass = search_target.sum(1)
                denominator = 1.0 + effective_eta * target_mass
                bounded_target = (
                    base_p_legal
                    + effective_eta.unsqueeze(1) * search_target
                ) / denominator.unsqueeze(1)
                bounded_target_logp = bounded_target.clamp_min(1e-30).log()
                bounded_action = bounded_target.argmax(1)
                active = measured & (effective_eta > 0)
                bounded_all_action_changes += int(
                    ((bounded_action != ba) & active).sum())
                bounded_all_teacher_matches += int(
                    ((bounded_action == hard_target) & active).sum())
                bounded_all_n += int(active.sum())

                residual_terms = bounded_target * (
                    bounded_target_logp - sl_legal_logp)
                target_base_terms = bounded_target * (
                    bounded_target_logp - bl_legal_logp)
                base_target_terms = base_p_legal * (
                    bl_legal_logp - bounded_target_logp)
                residual_terms = torch.where(
                    bounded_target > 0, residual_terms,
                    torch.zeros_like(residual_terms))
                target_base_terms = torch.where(
                    bounded_target > 0, target_base_terms,
                    torch.zeros_like(target_base_terms))
                base_target_terms = torch.where(
                    legal, base_target_terms,
                    torch.zeros_like(base_target_terms))
                if active.any():
                    bounded_all_residual_kls.append(
                        residual_terms.sum(1)[active].cpu())
                    bounded_all_target_base_kls.append(
                        target_base_terms.sum(1)[active].cpu())
                    bounded_all_base_target_kls.append(
                        base_target_terms.sum(1)[active].cpu())

                active_edit = edit & active
                bounded_opt_adopt += int(
                    ((bounded_action == hard_target) & active_edit).sum())
                bounded_opt_edit_n += int(active_edit.sum())
                if active_edit.any():
                    bounded_residual_kls.append(
                        residual_terms.sum(1)[active_edit].cpu())
                    bounded_target_base_kls.append(
                        target_base_terms.sum(1)[active_edit].cpu())
                    edit_rows = torch.arange(len(hard_target), device=device)
                    edit_teacher_prob_before += float(
                        teacher_p[active_edit].sum())
                    edit_teacher_prob_after += float(
                        sl_legal_logp.exp()[
                            edit_rows[active_edit],
                            hard_target[active_edit]].sum())
            n += len(obs)
        values = torch.cat(kls).numpy() if kls else np.array([])
        legal_values = (torch.cat(legal_kls).numpy()
                        if legal_kls else np.array([]))
        report = {
            'n': n, 'retain': retains / max(n, 1),
            'mean_kl': float(values.mean()) if len(values) else float('nan'),
            'p90_kl': float(np.percentile(values, 90))
            if len(values) else float('nan'),
            'mean_legal_kl': (float(legal_values.mean())
                              if len(legal_values) else float('nan')),
            'p90_legal_kl': (float(np.percentile(legal_values, 90))
                             if len(legal_values) else float('nan')),
            'teacher_before': teacher_before / max(teacher_n, 1),
            'teacher_after': teacher_after / max(teacher_n, 1),
            'teacher_n': teacher_n,
            'edit_adopt': edit_adopt / max(edit_n, 1),
            'edit_n': edit_n,
            'agree_preserve': agree_preserve / max(agree_n, 1),
        }
        if bounded_eta is not None and name.startswith('target'):
            gaps = (torch.cat(edit_probability_gaps).numpy()
                    if edit_probability_gaps else np.array([]))
            residual = (torch.cat(bounded_residual_kls).numpy()
                        if bounded_residual_kls else np.array([]))
            target_base = (torch.cat(bounded_target_base_kls).numpy()
                           if bounded_target_base_kls else np.array([]))
            all_residual = (torch.cat(bounded_all_residual_kls).numpy()
                            if bounded_all_residual_kls else np.array([]))
            all_target_base = (
                torch.cat(bounded_all_target_base_kls).numpy()
                if bounded_all_target_base_kls else np.array([]))
            all_base_target = (
                torch.cat(bounded_all_base_target_kls).numpy()
                if bounded_all_base_target_kls else np.array([]))
            report.update({
                'bounded_eta': float(bounded_eta),
                'bounded_soft_alpha': float(bounded_soft_alpha),
                'bounded_source_weights': bounded_source_weights,
                'bounded_agree_weight': float(bounded_agree_weight),
                'bounded_disagree_weight': float(bounded_disagree_weight),
                'bounded_optimum_all_action_change': (
                    bounded_all_action_changes / max(bounded_all_n, 1)),
                'bounded_optimum_all_teacher_match': (
                    bounded_all_teacher_matches / max(bounded_all_n, 1)),
                'bounded_optimum_all_n': bounded_all_n,
                'bounded_optimum_all_residual_kl': (
                    float(all_residual.mean())
                    if len(all_residual) else float('nan')),
                'bounded_optimum_all_base_kl': (
                    float(all_target_base.mean())
                    if len(all_target_base) else float('nan')),
                'bounded_optimum_all_base_to_target_kl': (
                    float(all_base_target.mean())
                    if len(all_base_target) else float('nan')),
                'bounded_optimum_edit_adopt': (
                    bounded_opt_adopt / max(bounded_opt_edit_n, 1)),
                'bounded_optimum_edit_n': bounded_opt_edit_n,
                'edit_probability_gap_p50': (
                    float(np.percentile(gaps, 50))
                    if len(gaps) else float('nan')),
                'edit_probability_gap_p90': (
                    float(np.percentile(gaps, 90))
                    if len(gaps) else float('nan')),
                'edit_teacher_probability_before': (
                    edit_teacher_prob_before / max(bounded_opt_edit_n, 1)),
                'edit_teacher_probability_after': (
                    edit_teacher_prob_after / max(bounded_opt_edit_n, 1)),
                'bounded_optimum_residual_kl': (
                    float(residual.mean())
                    if len(residual) else float('nan')),
                'bounded_optimum_base_kl': (
                    float(target_base.mean())
                    if len(target_base) else float('nan')),
            })
        out[name] = report
    student.train(was_training)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--targets', required=True)
    p.add_argument('--anchors', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--base-policy-sidecar',
                   help='Exact original-view base/teacher disagreement .npz '
                        'from add_fulllegal_mask. Required for disagreement '
                        'weighting together with D4 augmentation.')
    p.add_argument(
        '--sidecar-disagree-key', default='disagree',
        choices=('disagree', 'recorded_disagree'),
        help=('Sidecar edit mask. recorded_disagree uses the actor-root '
              'clean-prior argmax when the corpus preserved base_move; '
              'disagree uses a separate deployment-style batch recompute.'))
    p.add_argument(
        '--edit-weight-sidecar',
        help=('Optional corpus-matched sidecar whose mask defines high-dose '
              'teacher rows. The base sidecar still must match --base. This '
              'allows architecture controls to weight the same teacher-edit '
              'rows without pretending they are candidate-base edits.'))
    p.add_argument(
        '--edit-weight-key', choices=('disagree', 'recorded_disagree'),
        help=('Mask in --edit-weight-sidecar. Defaults to '
              '--sidecar-disagree-key when no separate sidecar is used.'))
    p.add_argument('--objective', choices=['bounded', 'aggregate', 'legacy'],
                   default='bounded')
    p.add_argument('--eta', type=float, default=0.1,
                   help='Search/base ratio for bounded objective.')
    p.add_argument('--soft-alpha', type=float, default=1.0,
                   help='1=raw visits, 0=teacher hard, .5=50/50 mix.')
    p.add_argument('--source-weights', type=float, nargs='+',
                   help='Optional target multiplier per source_id, in '
                        'manifest source order. All rows remain present.')
    p.add_argument('--base-agree-weight', type=float, default=1.0,
                   help='Target multiplier when frozen-base and teacher '
                        'legal argmax agree.')
    p.add_argument('--base-disagree-weight', type=float, default=1.0,
                   help='Target multiplier when frozen-base and teacher '
                        'legal argmax disagree.')
    p.add_argument(
        '--edit-exposure', type=float, default=1.0,
        help=('Bounded objective only: relative whole-row weight for labelled '
              'teacher/base disagreements. This multiplies both base KL and '
              'teacher CE, increasing optimization exposure without changing '
              'the row\'s exact function-space target.'))
    p.add_argument('--update-bn', action='store_true',
                   help='Aggregate/legacy only: update train-mode BatchNorm. '
                        'Default is frozen deployment-mode BN.')
    p.add_argument('--anchor-weight', type=float, default=1.0,
                   help='Anchor loss multiplier. Aggregate uses base KL; '
                        'legacy uses recorded greedy hard CE.')
    p.add_argument('--from-scratch', action='store_true',
                   help='Reset the network for aggregate/legacy policy '
                        'distillation. Requires --update-bn.')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument(
        '--stop-after-epoch', type=int, default=0,
        help=('Operational gate: stop cleanly after saving this completed '
              'epoch while retaining the scheduler implied by --epochs. '
              'May be changed when resuming. 0 runs through --epochs.'))
    p.add_argument('--gate-min-retain', type=float, default=0.0,
                   help='Stop after an epoch whose minimum deployment-FP '
                        'audit retention falls below this value. 0 disables.')
    p.add_argument('--gate-max-legal-kl', type=float, default=-1.0,
                   help='Stop after an epoch whose maximum mean deployment '
                        'legal KL exceeds this value. Negative disables.')
    p.add_argument('--gate-max-new-bn-nonfinite', type=int, default=-1,
                   help='Stop when deployment casting creates more than this '
                        'many BN running-var nonfinites beyond the frozen '
                        'base. Negative disables.')
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--schedule', choices=['cosine', 'constant'],
                   default='cosine')
    p.add_argument('--warmup-fraction', type=float, default=0.05,
                   help='Fraction of optimizer steps used for 0.1x->1x LR.')
    p.add_argument('--min-lr-ratio', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=20260808)
    p.add_argument('--device')
    p.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'],
                   default='fp16',
                   help='Forward precision; losses and KL remain float32.')
    p.add_argument('--legal-support-loss', action='store_true',
                   help='Normalize CE/KL only over moves the deployment '
                        'engine can legally choose. Exact legality is rebuilt '
                        'from each (possibly augmented) observation.')
    p.add_argument('--no-dihedral-augment', action='store_true')
    p.add_argument('--color-augment', action='store_true')
    p.add_argument('--augment-factor', type=int, default=1,
                   help='Random symmetry views per base row per epoch.')
    p.add_argument('--save-every-steps', type=int, default=250)
    p.add_argument('--resume-every-steps', type=int, default=0,
                   help='Atomically overwrite latest.pt this often during an '
                        'epoch. 0 uses --save-every-steps.')
    p.add_argument('--archive-every-epochs', type=int, default=1,
                   help='Write model-only epoch_N checkpoints at this '
                        'interval (and always at the final epoch). latest.pt '
                        'is atomically overwritten with a resumable checkpoint '
                        'after every epoch. 0 disables intermediate archives.')
    p.add_argument('--resume',
                   help='Resume an epoch-boundary latest.pt (or an older '
                        'epoch_N checkpoint that contains optimizer state). '
                        'Optimization-defining arguments must match exactly.')
    p.add_argument('--audit-batches', type=int, default=20)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--max-target-states', type=int, default=0)
    p.add_argument('--max-anchor-states', type=int, default=0)
    p.add_argument(
        '--min-target-edits', type=int, default=0,
        help=('When --max-target-states caps a smoke, reserve up to this many '
              'declared edit rows before sampling preservation rows. A value '
              'larger than the available training edits includes them all.'))
    p.add_argument('--save-dir', default='checkpoints/flywheel')
    args = p.parse_args()

    if not 0 <= args.soft_alpha <= 1:
        raise ValueError('--soft-alpha must be in [0,1]')
    if args.objective == 'bounded' and args.eta <= 0:
        raise ValueError('--eta must be positive for bounded objective')
    if args.base_agree_weight < 0 or args.base_disagree_weight < 0:
        raise ValueError('base agree/disagree weights must be nonnegative')
    if args.edit_exposure <= 0:
        raise ValueError('--edit-exposure must be positive')
    if args.objective != 'bounded' and args.edit_exposure != 1:
        raise ValueError('--edit-exposure is only defined for bounded '
                         'objective')
    if args.update_bn and args.objective not in ('aggregate', 'legacy'):
        raise ValueError('--update-bn is only defined for aggregate or '
                         'legacy objectives')
    if args.anchor_weight < 0:
        raise ValueError('--anchor-weight must be nonnegative')
    if args.objective == 'bounded' and args.anchor_weight != 1:
        raise ValueError('--anchor-weight is not used by bounded objective')
    if args.from_scratch and args.objective == 'bounded':
        raise ValueError('--from-scratch needs aggregate or legacy objective')
    if args.from_scratch and not args.update_bn:
        raise ValueError('--from-scratch requires --update-bn; frozen random '
                         'BatchNorm statistics are not a deployment model')
    if args.augment_factor < 1:
        raise ValueError('--augment-factor must be >=1')
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError('--warmup-fraction must be in [0,1)')
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError('--min-lr-ratio must be in [0,1]')
    if args.archive_every_epochs < 0:
        raise ValueError('--archive-every-epochs must be >=0')
    if args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs:
        raise ValueError('--stop-after-epoch must be 0 or in [1, --epochs]')
    if not 0 <= args.gate_min_retain <= 1:
        raise ValueError('--gate-min-retain must be in [0,1]')
    if args.gate_max_new_bn_nonfinite < -1:
        raise ValueError('--gate-max-new-bn-nonfinite must be >=-1')
    if args.save_every_steps < 0 or args.resume_every_steps < 0:
        raise ValueError('step checkpoint intervals must be nonnegative')
    if args.min_target_edits < 0:
        raise ValueError('--min-target-edits must be nonnegative')
    if (args.min_target_edits and args.max_target_states
            and args.min_target_edits > args.max_target_states):
        raise ValueError('--min-target-edits cannot exceed '
                         '--max-target-states')
    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(args.resume)
    if args.edit_weight_sidecar and not args.base_policy_sidecar:
        raise ValueError('--edit-weight-sidecar requires '
                         '--base-policy-sidecar for base provenance')
    if args.edit_weight_key and not args.edit_weight_sidecar:
        raise ValueError('--edit-weight-key requires --edit-weight-sidecar')
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False)
        if not isinstance(resume_checkpoint, dict):
            raise ValueError('--resume needs a flywheel checkpoint dict')
        _validate_resume_args(
            resume_checkpoint.get('args', {}), vars(args))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    loader_rng = torch.Generator().manual_seed(args.seed)
    if args.precision == 'fp32' or device.type == 'cpu':
        amp_dtype = torch.float32
    elif args.precision == 'bf16':
        if device.type != 'cuda' or not torch.cuda.is_bf16_supported():
            raise ValueError('bf16 requires a CUDA device with bf16 support')
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler = (torch.amp.GradScaler('cuda')
              if device.type == 'cuda' and amp_dtype == torch.float16
              else None)
    print(f'device={device}; objective={args.objective}; eta={args.eta:g}; '
          f'soft_alpha={args.soft_alpha:g}; precision={amp_dtype}; '
          f'loss_support={"legal" if args.legal_support_loss else "full"}; '
          f'BN={"update" if args.update_bn else "frozen"}; '
          f'init={"resume" if args.resume else ("scratch" if args.from_scratch else "base")}',
          flush=True)

    aug = not args.no_dihedral_augment
    target_train = make_datasets(
        args.targets, device, augment=aug,
        color_augment=args.color_augment,
        augment_factor=args.augment_factor, train=True)
    anchor_train = make_datasets(
        args.anchors, device, augment=aug,
        color_augment=args.color_augment,
        augment_factor=args.augment_factor, train=True)
    target_val = make_datasets(
        args.targets, device, augment=False, color_augment=False,
        augment_factor=1, train=False)
    anchor_val = make_datasets(
        args.anchors, device, augment=False, color_augment=False,
        augment_factor=1, train=False)
    if args.base_policy_sidecar:
        sidecar = np.load(args.base_policy_sidecar, allow_pickle=False)
        if 'metadata' in sidecar:
            sidecar_metadata = json.loads(str(sidecar['metadata']))
            recorded_base_hash = sidecar_metadata.get('base_sha256')
            if (recorded_base_hash
                    and recorded_base_hash != _sha256(args.base)):
                raise ValueError(
                    'base-policy sidecar checkpoint hash does not match '
                    f'{args.base!r}')
            corpus_md = target_train.metadata or {}
            expected_inventory = corpus_md.get('input_inventory_sha256')
            recorded_inventory = sidecar_metadata.get(
                'tensor_inventory_sha256')
            if (expected_inventory and recorded_inventory
                    and expected_inventory != recorded_inventory):
                raise ValueError('base-policy sidecar corpus inventory does '
                                 'not match target tensor')
        weight_sidecar = sidecar
        weight_path = args.base_policy_sidecar
        weight_key = args.sidecar_disagree_key
        if args.edit_weight_sidecar:
            weight_sidecar = np.load(
                args.edit_weight_sidecar, allow_pickle=False)
            weight_path = args.edit_weight_sidecar
            weight_key = args.edit_weight_key or args.sidecar_disagree_key
            if 'metadata' in weight_sidecar:
                weight_metadata = json.loads(str(weight_sidecar['metadata']))
                expected_inventory = (target_train.metadata or {}).get(
                    'input_inventory_sha256')
                recorded_inventory = weight_metadata.get(
                    'tensor_inventory_sha256')
                if (expected_inventory and recorded_inventory
                        and expected_inventory != recorded_inventory):
                    raise ValueError('edit-weight sidecar corpus inventory '
                                     'does not match target tensor')
        if weight_key not in weight_sidecar:
            raise ValueError(
                f'edit-weight sidecar has no {weight_key!r} field')
        sidecar_disagree = weight_sidecar[weight_key]
        if len(sidecar_disagree) != len(target_train.boards):
            raise ValueError('edit-weight sidecar row count does not match '
                             'target tensor')
        original_disagree = torch.from_numpy(
            sidecar_disagree.astype(bool)).to(device)
        anchor_not_edit = torch.zeros(
            len(anchor_train.boards), dtype=torch.bool, device=device)
        target_train.flywheel_disagree = original_disagree
        target_val.flywheel_disagree = original_disagree
        anchor_train.flywheel_disagree = anchor_not_edit
        anchor_val.flywheel_disagree = anchor_not_edit
        print(f'edit-weight sidecar {weight_path}[{weight_key}]: '
              f'{int(original_disagree.sum()):,}/'
              f'{len(original_disagree):,} original-view disagreements',
              flush=True)
    elif (not args.no_dihedral_augment
          and args.base_agree_weight != args.base_disagree_weight):
        raise ValueError('D4 plus unequal base agreement weights requires '
                         '--base-policy-sidecar so transformed non-equivariance '
                         'is not mislabeled as a search edit')
    target_names = (target_train.metadata or {}).get('source_names', [])
    n_sources = len(target_names)
    if args.source_weights is not None:
        if len(args.source_weights) != n_sources:
            raise ValueError('--source-weights needs one value per target '
                             f'source ({n_sources}: {target_names})')
        if any(w < 0 for w in args.source_weights):
            raise ValueError('--source-weights must be nonnegative')
        source_weights = torch.tensor(
            args.source_weights, dtype=torch.float32, device=device)
    else:
        source_weights = torch.ones(n_sources, device=device)
    print('target source weights: ' + ', '.join(
        f'{name}={float(source_weights[i]):g}'
        for i, name in enumerate(target_names)), flush=True)
    cap_dataset(target_train, args.max_target_states, rng,
                args.min_target_edits)
    cap_dataset(anchor_train, args.max_anchor_states, rng)
    # Functional audits use one fixed random subset rather than the prefix of
    # a source-ordered artifact.  This makes checkpoint-to-checkpoint changes
    # comparable and keeps long-game rows from one early file dominating.
    audit_rows = args.audit_batches * args.batch_size
    if audit_rows > 0:
        audit_rng = np.random.default_rng(args.seed + 1)
        cap_dataset(target_val, audit_rows, audit_rng)
        cap_dataset(anchor_val, audit_rows, audit_rng)

    def loader(ds, shuffle):
        return TensorBatchLoader(ds, args.batch_size, shuffle=shuffle,
                                 generator=loader_rng)

    train_loader = PooledBatchLoader(
        target_train, anchor_train, args.batch_size,
        shuffle=True, generator=loader_rng)
    val_loaders = {'target': loader(target_val, False),
                   'anchor': loader(anchor_val, False)}
    print(f'train rows: target={len(target_train):,}; '
          f'anchor={len(anchor_train):,}; ratio='
          f'{len(anchor_train)/max(len(target_train),1):.2f}:1', flush=True)

    loaded, _ = load_model(args.base, device, fp16=False)
    loaded.train(False)
    loaded.requires_grad_(False)
    student = copy.deepcopy(loaded)
    if resume_checkpoint is not None:
        resume_state = resume_checkpoint.get('model')
        if resume_state is None:
            raise ValueError('resume checkpoint has no model state')
        if any(k.startswith('_orig_mod.') for k in resume_state):
            resume_state = {
                k.replace('_orig_mod.', ''): value
                for k, value in resume_state.items()}
        student.load_state_dict(resume_state, strict=True)
    elif args.from_scratch:
        # Use each PyTorch module's native initializer. BatchNorm's reset also
        # restores its running mean and variance before train-mode updates.
        def reset_module(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        student.apply(reset_module)
    student.requires_grad_(True)
    # There is no dropout in PolicyNet.  Eval mode is deliberate for bounded
    # and frozen legacy runs: affine BN parameters and all weights still receive
    # gradients, while normalization remains exactly deployed.  The explicit
    # legacy control can instead reproduce train-mode BN updates.
    student.train(args.update_bn)
    if not args.update_bn:
        _quantize_frozen_bn_for_deployment_(student, amp_dtype)
    # Anchor to the function which actually generated/plays the corpus. Native
    # MPS inference casts weights and BatchNorm buffers to fp16; fp32 autocast
    # is not equivalent when a running variance overflows the fp16 range.
    base = _deployment_copy(loaded, amp_dtype)
    del loaded
    base_dtype = next(base.parameters()).dtype
    base_deployment_numerics = _deployment_numerics(base, amp_dtype)
    student_state = student.state_dict()
    base_state = {
        key: value.detach().to(dtype=student_state[key].dtype).clone()
        for key, value in base.state_dict().items()
    }
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(round(total_steps * args.warmup_fraction))

    def lr_multiplier(step):
        if args.schedule == 'constant':
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return 0.1 + 0.9 * step / warmup_steps
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_multiplier)
    os.makedirs(args.save_dir, exist_ok=True)
    run_manifest_path = os.path.join(args.save_dir, 'run_manifest.json')
    run_manifest = {
        'schema_version': 1,
        'args': {key: value for key, value in vars(args).items()
                 if key != 'resume'},
        'rows': {'target': len(target_train), 'anchor': len(anchor_train),
                 'batches_per_epoch': len(train_loader),
                 'total_steps': total_steps},
        'model_parameters': sum(p.numel() for p in student.parameters()),
        'deployment_protocol': {
            'dtype': str(amp_dtype),
            'base_numerics': base_deployment_numerics,
            'frozen_bn_buffers_quantized': not args.update_bn,
        },
        'corpus_metadata': {'targets': target_train.metadata,
                            'anchors': anchor_train.metadata},
    }
    if not os.path.exists(run_manifest_path):
        _atomic_json_save(run_manifest, run_manifest_path)

    completed_epochs = 0
    global_step = 0
    active_epoch = 0
    resume_batch = 0
    resume_sums = None
    if resume_checkpoint is not None:
        if 'optimizer' not in resume_checkpoint or 'scheduler' not in resume_checkpoint:
            raise ValueError('resume checkpoint is model-only; use latest.pt '
                             'or a resumable epoch checkpoint')
        opt.load_state_dict(resume_checkpoint['optimizer'])
        scheduler.load_state_dict(resume_checkpoint['scheduler'])
        if scaler is not None and resume_checkpoint.get('scaler') is not None:
            scaler.load_state_dict(resume_checkpoint['scaler'])
        completed_epochs = _completed_epoch(resume_checkpoint, args.resume)
        global_step = int(resume_checkpoint.get('global_step', -1))
        active_epoch = int(resume_checkpoint.get('active_epoch') or 0)
        resume_batch = int(resume_checkpoint.get('batch_in_epoch') or 0)
        resume_sums = resume_checkpoint.get('epoch_sums')
        if active_epoch:
            if completed_epochs != active_epoch - 1:
                raise ValueError(
                    'mid-epoch checkpoint has inconsistent completed/active '
                    f'epochs: {completed_epochs}/{active_epoch}')
            if not 0 < resume_batch < len(train_loader):
                raise ValueError(
                    f'invalid mid-epoch batch {resume_batch}/'
                    f'{len(train_loader)}')
            expected_step = completed_epochs * len(train_loader) + resume_batch
        else:
            expected_step = completed_epochs * len(train_loader)
        if global_step != expected_step:
            raise ValueError(
                f'resume step={global_step}, expected {expected_step} from '
                f'epoch/batch progress')
        exact_rng = _restore_rng_state(
            resume_checkpoint.get('rng_state'), loader_rng, rng, device)
        position = (f'active_epoch={active_epoch} batch={resume_batch}'
                    if active_epoch else f'completed_epoch={completed_epochs}')
        print(f'resumed {args.resume}: {position} step={global_step}; '
              f'rng={"exact" if exact_rng else "reset"}', flush=True)
        if not active_epoch and completed_epochs >= args.epochs:
            raise ValueError(f'checkpoint already completed {completed_epochs} '
                             f'of {args.epochs} epochs')

    def save(tag, *, resumable=False, completed_epoch=None,
             active_epoch=None, batch_in_epoch=0, epoch_sums=None):
        sd = student.state_dict()
        if not args.update_bn:
            for key, value in sd.items():
                if (key.endswith('.running_mean')
                        or key.endswith('.running_var')):
                    if not torch.equal(value, base_state[key]):
                        raise AssertionError(f'BN buffer drifted: {key}')
        path = os.path.join(args.save_dir, f'{tag}.pt')
        payload = {'model': sd, 'policy_only': True, 'args': vars(args),
                   'global_step': global_step,
                   'completed_epoch': completed_epoch,
                   'active_epoch': active_epoch,
                   'batch_in_epoch': batch_in_epoch,
                   'epoch_sums': epoch_sums,
                   'corpus_metadata': {
                       'targets': target_train.metadata,
                       'anchors': anchor_train.metadata}}
        if resumable:
            payload.update({'optimizer': opt.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'scaler': (scaler.state_dict()
                                       if scaler is not None else None),
                            'rng_state': _capture_rng_state(
                                loader_rng, rng, device)})
        _atomic_torch_save(payload, path)
        if resumable:
            _atomic_json_save({
                'schema_version': 1, 'checkpoint': path,
                'global_step': global_step,
                'completed_epoch': completed_epoch,
                'active_epoch': active_epoch,
                'batch_in_epoch': batch_in_epoch,
                'epoch_sums': epoch_sums,
            }, os.path.join(args.save_dir, 'progress.json'))
        print(f'saved {path}', flush=True)

    first_epoch = active_epoch or (completed_epochs + 1)
    resume_every_steps = (args.resume_every_steps
                          or args.save_every_steps)
    for epoch in range(first_epoch, args.epochs + 1):
        # Consume every target and anchor exactly once, mixed within batches.
        # This is the natural all-data objective under Adam, rather than a
        # high-variance alternation of correction and pullback updates.
        student.train(args.update_bn)
        start_batch = resume_batch if epoch == active_epoch else 0
        sums = (dict(resume_sums) if start_batch and resume_sums else {
            'loss': 0.0, 'kl': 0.0, 'nll': 0.0, 'grad_norm': 0.0,
            'rows': 0, 'target_rows': 0, 'teacher_rows': 0,
            'teacher_disagree': 0, 'online_teacher_disagree': 0,
            'student_teacher_match': 0, 'student_base_retain': 0,
            'edit_adopt': 0, 'exposure_mass': 0.0,
            'exposed_edit_rows': 0,
            'source_rows': [0] * n_sources,
            'source_weight': [0.0] * n_sources,
        })
        t0 = time.time()
        epoch_seed = args.seed + epoch * 1_000_003
        batches = train_loader.iter_from(
            start_batch=start_batch, epoch_seed=epoch_seed)
        for si, batch in enumerate(batches, start_batch + 1):
            if len(batch) == 8:
                (obs, policy, target_weight, behavior, hard_target,
                 source_id, recorded_disagree, is_target) = batch
            else:
                (obs, policy, target_weight, behavior, hard_target,
                 source_id, is_target) = batch
                recorded_disagree = None
            obs = obs.to(device, non_blocking=True)
            legal_mask = (legal_mask_from_observation(obs)
                          if args.legal_support_loss else None)
            if legal_mask is not None:
                # Recorded C++ moves are engine-validated legal.  The batched
                # GPU component labeller disagrees on a tiny winding-corridor
                # tail (~0.008% in a 50k anchor audit), so retain all recorded
                # search support rather than turn those rows into infinite CE.
                # Ordinary off-label illegal logits remain fully excluded.
                legal_mask = legal_mask | (policy > 0)
                legal_mask.scatter_(
                    1, hard_target.long().unsqueeze(1), True)
            target_weight = target_weight.clone()
            label_available = target_weight > 0
            row_exposure = torch.ones_like(target_weight, dtype=torch.float32)
            if is_target.any():
                target_weight[is_target] *= source_weights[
                    source_id[is_target].long()]
            use_amp = amp_dtype != torch.float32
            need_base = (
                args.objective in ('bounded', 'aggregate')
                or args.base_agree_weight != args.base_disagree_weight)
            if need_base:
                with torch.inference_mode():
                    b = base(obs.to(dtype=base_dtype))
                    base_logits = b[0] if isinstance(b, tuple) else b
            with torch.amp.autocast(device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                s = student(obs)
                student_logits = s[0] if isinstance(s, tuple) else s
                if not need_base:
                    # The legacy loss never reads base probabilities.  Avoid a
                    # redundant frozen-model forward in this control arm.
                    base_logits = student_logits.detach()
            teacher_disagree = online_disagree = None
            student_action = base_action = None
            if need_base:
                base_for_argmax = base_logits.float()
                argmax_legal = (legal_mask if legal_mask is not None
                                else legal_mask_from_observation(obs))
                base_for_argmax = base_for_argmax.masked_fill(
                    ~argmax_legal, float('-inf'))
                base_action = base_for_argmax.argmax(1)
                student_action = student_logits.float().masked_fill(
                    ~argmax_legal, float('-inf')).argmax(1)
                online_disagree = base_action != hard_target.long()
                teacher_disagree = (recorded_disagree.bool()
                                    if recorded_disagree is not None
                                    else online_disagree)
                edit_weight = (
                    teacher_disagree.float() * args.base_disagree_weight
                    + (~teacher_disagree).float() * args.base_agree_weight)
                target_weight[is_target] *= edit_weight[is_target]
                exposed_edit = (teacher_disagree & is_target
                                & label_available)
                row_exposure[exposed_edit] = args.edit_exposure
            loss, kl, nll = flywheel_loss(
                base_logits, student_logits, policy, hard_target,
                target_weight,
                objective=args.objective, eta=args.eta,
                soft_alpha=args.soft_alpha, is_target=is_target,
                anchor_weight=args.anchor_weight, legal_mask=legal_mask,
                row_weight=row_exposure)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(), 1.0)
                opt.step()
            scheduler.step()
            global_step += 1

            nr = len(obs)
            sums['loss'] += float(loss.detach()) * nr
            sums['kl'] += float(kl.sum())
            sums['grad_norm'] += float(grad_norm) * nr
            sums['exposure_mass'] += float(row_exposure.sum())
            sums['exposed_edit_rows'] += int(
                ((row_exposure != 1) & is_target).sum())
            n_target = int(is_target.sum())
            if n_target:
                sums['nll'] += float(
                    (target_weight[is_target] * nll[is_target]).sum())
                sums['target_rows'] += float(target_weight[is_target].sum())
                sums['teacher_rows'] += n_target
                source_counts = torch.bincount(
                    source_id[is_target].long(), minlength=n_sources)
                source_weight_sums = torch.zeros(
                    n_sources, device=device).scatter_add_(
                        0, source_id[is_target].long(),
                        target_weight[is_target].float())
                for sid in range(n_sources):
                    sums['source_rows'][sid] += int(source_counts[sid])
                    sums['source_weight'][sid] += float(
                        source_weight_sums[sid])
                if teacher_disagree is not None:
                    target_edit = teacher_disagree & is_target
                    sums['teacher_disagree'] += int(target_edit.sum())
                    sums['online_teacher_disagree'] += int(
                        (online_disagree & is_target).sum())
                    sums['student_teacher_match'] += int((
                        (student_action == hard_target.long())
                        & is_target).sum())
                    sums['student_base_retain'] += int((
                        (student_action == base_action) & is_target).sum())
                    sums['edit_adopt'] += int((
                        (student_action == hard_target.long())
                        & target_edit).sum())
            sums['rows'] += nr
            if (args.save_every_steps
                    and global_step % args.save_every_steps == 0):
                save(f'e{epoch}_s{global_step}', completed_epoch=epoch - 1)
            if (resume_every_steps
                    and global_step % resume_every_steps == 0):
                if si < len(train_loader):
                    save('latest', resumable=True,
                         completed_epoch=epoch - 1,
                         active_epoch=epoch, batch_in_epoch=si,
                         epoch_sums=sums)
            if si % args.log_every == 0 or si == len(train_loader):
                elapsed = time.time() - t0
                batches_this_run = si - start_batch
                eta_s = (elapsed / max(batches_this_run, 1)
                         * (len(train_loader) - si))
                disagree_rate = (sums['teacher_disagree']
                                 / max(sums['teacher_rows'], 1))
                online_disagree_rate = (sums['online_teacher_disagree']
                                        / max(sums['teacher_rows'], 1))
                edit_adopt = (sums['edit_adopt']
                              / max(sums['teacher_disagree'], 1))
                print(f'  ep{epoch} {si}/{len(train_loader)} step={global_step} '
                      f'lr={opt.param_groups[0]["lr"]:.2e} '
                      f'loss={sums["loss"]/sums["rows"]:.5f} '
                      f'KL={sums["kl"]/sums["rows"]:.5f} '
                      f'grad={sums["grad_norm"]/sums["rows"]:.3f} '
                      f'targetNLL={sums["nll"]/max(sums["target_rows"],1):.3f} '
                      f'teacherDisagree={disagree_rate:.3f} '
                      f'onlineAugDisagree={online_disagree_rate:.3f} '
                      f'exposureMass={sums["exposure_mass"]/sums["rows"]:.3f} '
                      f'editAdopt={edit_adopt:.3f} '
                      f'ETA={eta_s:.0f}s', flush=True)

        deployment_numerics = _deployment_numerics(student, amp_dtype)
        audit_student = (student if amp_dtype == torch.float32
                         else _deployment_copy(student, amp_dtype))
        metrics = audit(
            audit_student, base, val_loaders, device,
            max_batches=args.audit_batches, amp_dtype=amp_dtype,
            bounded_eta=(args.eta if args.objective == 'bounded' else None),
            bounded_source_weights=args.source_weights,
            bounded_agree_weight=args.base_agree_weight,
            bounded_disagree_weight=args.base_disagree_weight,
            bounded_soft_alpha=args.soft_alpha)
        if audit_student is not student:
            del audit_student
            if device.type == 'mps':
                torch.mps.empty_cache()
        print(f'[epoch {epoch}] ' + '; '.join(
            f'{name}: retain={v["retain"]:.3f} '
            f'KLfull={v["mean_kl"]:.5f}/{v["p90_kl"]:.5f} '
            f'KLlegal={v["mean_legal_kl"]:.5f}/'
            f'{v["p90_legal_kl"]:.5f} '
            f'teacher={v["teacher_before"]:.3f}->'
            f'{v["teacher_after"]:.3f} '
            f'editAdopt={v["edit_adopt"]:.3f}/{v["edit_n"]}'
            + (f' ideal@eta={v["bounded_optimum_edit_adopt"]:.3f}'
               f' qKLedit={v["bounded_optimum_residual_kl"]:.4f}'
               f' qKLall={v["bounded_optimum_all_residual_kl"]:.4f}'
               if 'bounded_optimum_edit_adopt' in v else '')
            for name, v in metrics.items())
            + f'; deployBNvarNonfinite='
              f'{deployment_numerics["bn_running_var_cast_nonfinite"]}',
            flush=True)
        functional_gate = _functional_gate(
            metrics, deployment_numerics, base_deployment_numerics,
            min_retain=args.gate_min_retain,
            max_legal_kl=args.gate_max_legal_kl,
            max_new_bn_nonfinite=args.gate_max_new_bn_nonfinite)
        epoch_report = {
            'schema_version': 1, 'epoch': epoch,
            'global_step': global_step,
            'lr': opt.param_groups[0]['lr'],
            'train': sums, 'audit': metrics,
            'audit_protocol': {
                'deployment_dtype': str(amp_dtype),
                'true_module_cast': amp_dtype != torch.float32,
            },
            'deployment_numerics': deployment_numerics,
            'functional_gate': functional_gate,
        }
        _atomic_json_save(
            epoch_report,
            os.path.join(args.save_dir, f'diagnostics_epoch_{epoch}.json'))
        # Keep one crash-safe resumable checkpoint without accumulating an
        # optimizer-sized file every epoch. Model-only archives are sufficient
        # for functional/gameplay evaluation and remain much smaller.
        save('latest', resumable=True, completed_epoch=epoch)
        if ((args.archive_every_epochs
             and epoch % args.archive_every_epochs == 0)
                or epoch == args.epochs):
            save(f'epoch_{epoch}', completed_epoch=epoch)
        if functional_gate['failures']:
            print('operational safety gate stopped after completed epoch '
                  f'{epoch}: ' + '; '.join(functional_gate['failures'])
                  + f'; resume from {args.save_dir}/latest.pt', flush=True)
            break
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            print(f'operational stop after completed epoch {epoch}; '
                  f'resume from {args.save_dir}/latest.pt', flush=True)
            break
        # Only the first resumed epoch can begin partway through an epoch.
        active_epoch = resume_batch = 0
        resume_sums = None


if __name__ == '__main__':
    main()
