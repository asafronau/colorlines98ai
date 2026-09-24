"""Measure D4 and color-permutation action consistency on fixed states.

For each state, infer once in its recorded orientation and once under each of
the seven non-identity rotations/reflections.  Map every transformed legal
argmax back to the original coordinates and report agreement.  Optionally do
the same for random permutations of the seven color labels (actions need no
mapping in that case).  Both are exact game symmetries, so disagreement is a
representational/sample-efficiency tax, although it does not by itself prove a
gameplay error.

Run under caffeinate.  FP16 is the default protocol.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from alphatrain.dataset import (
    TensorDatasetGPU, _LINE_INV_LUTS, _OBS_INV_LUTS, _POL_INV_LUTS,
    _POL_LUTS, _transform_observation,
)
from alphatrain.evaluate import load_model
from alphatrain.scripts.fleet_gpu import gpu_legal_mask


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--n-samples', type=int, default=20_000)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=20260808)
    p.add_argument('--color-permutations', type=int, default=7,
                   help='Random color relabelings per state; 0 disables.')
    p.add_argument('--device', default='mps')
    p.add_argument('--fp32', action='store_true',
                   help='Diagnostic override; FP16 is the default protocol.')
    args = p.parse_args()

    device = torch.device(args.device)
    ds = TensorDatasetGPU(args.tensor, augment=False, color_augment=False,
                          augment_factor=1, device=str(device))
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(
        len(ds.base_indices),
        size=min(args.n_samples, len(ds.base_indices)), replace=False))
    color_maps = np.zeros(
        (len(selected), args.color_permutations, 8), dtype=np.int64)
    if args.color_permutations:
        color_maps[:, :, 1:] = (
            np.argsort(rng.random(
                (len(selected), args.color_permutations, 7)), axis=2) + 1)
    ds.base_indices = torch.from_numpy(selected).to(device)
    obs_inv = torch.as_tensor(
        np.stack(_OBS_INV_LUTS), device=device, dtype=torch.long)
    line_inv = torch.as_tensor(
        np.stack(_LINE_INV_LUTS), device=device, dtype=torch.long)
    pol_inv = torch.as_tensor(
        np.stack(_POL_INV_LUTS), device=device, dtype=torch.long)
    pol_fwd = torch.as_tensor(
        np.stack(_POL_LUTS), device=device, dtype=torch.long)
    n = len(selected)

    for path in args.models:
        model, _ = load_model(path, device, fp16=not args.fp32)
        dtype = next(model.parameters()).dtype
        agree = np.zeros(8, dtype=np.int64)
        all_consistent = 0
        teacher_matches = np.zeros(8, dtype=np.int64)
        ensemble_matches = 0
        ensemble_reference_agree = 0
        d4_js_sum = 0.0
        color_agree = np.zeros(args.color_permutations, dtype=np.int64)
        color_teacher_matches = np.zeros(
            args.color_permutations, dtype=np.int64)
        all_colors_consistent = 0
        color_ensemble_matches = 0
        color_ensemble_reference_agree = 0
        color_js_sum = 0.0
        seen = 0

        for start in range(0, n, args.batch_size):
            pos = torch.arange(start, min(start + args.batch_size, n))
            obs, policy = ds.collate(pos)
            recorded = policy.argmax(1).long()
            actual = ds.base_indices[pos.to(device)]
            legal = gpu_legal_mask(ds.boards[actual])
            mapped_actions = []
            mapped_probs = []
            for t in range(8):
                if t == 0:
                    transformed_obs = obs
                    transformed_legal = legal
                else:
                    transformed_obs = _transform_observation(
                        obs, obs_inv[t], line_inv[t])
                    transformed_legal = legal[:, pol_inv[t]]
                logits = model(transformed_obs.to(dtype))
                if isinstance(logits, tuple):
                    logits = logits[0]
                action_new = logits.float().masked_fill(
                    ~transformed_legal, float('-inf')).argmax(1)
                action_old = (action_new if t == 0
                              else pol_inv[t, action_new])
                mapped_actions.append(action_old)
                probs_new = torch.softmax(
                    logits.float().masked_fill(
                        ~transformed_legal, float('-inf')), dim=1)
                # pol_fwd[old] is the corresponding action index in the
                # transformed orientation, so indexing by it maps the complete
                # distribution back into original coordinates.
                probs_old = (probs_new if t == 0
                             else probs_new[:, pol_fwd[t]])
                mapped_probs.append(probs_old)
                teacher_matches[t] += int((action_old == recorded).sum())
            actions = torch.stack(mapped_actions, dim=1)
            reference = actions[:, :1]
            agree += (actions == reference).sum(0).cpu().numpy()
            all_consistent += int((actions == reference).all(1).sum())
            d4_mean = torch.stack(mapped_probs, dim=0).mean(0)
            d4_action = d4_mean.argmax(1)
            ensemble_matches += int((d4_action == recorded).sum())
            ensemble_reference_agree += int(
                (d4_action == reference[:, 0]).sum())
            # Generalized Jensen-Shannon divergence across the eight mapped
            # legal policies.  Zero means exact distributional equivariance.
            mean_h = -(d4_mean * d4_mean.clamp_min(1e-30).log()).sum(1)
            view_h = torch.stack([
                -(p * p.clamp_min(1e-30).log()).sum(1)
                for p in mapped_probs], dim=0).mean(0)
            d4_js_sum += float((mean_h - view_h).sum())
            del mapped_probs, d4_mean
            if args.color_permutations:
                color_actions = []
                color_probs = []
                for ci in range(args.color_permutations):
                    color_map = torch.as_tensor(
                        color_maps[start:start + len(pos), ci],
                        device=device, dtype=torch.long)
                    # Channels 0:7 are one plane per old color.  Move those
                    # planes to their relabeled color slots using new->old.
                    color_inv = color_map.argsort(1)
                    color_obs = obs.clone()
                    gather_planes = (color_inv[:, 1:] - 1).reshape(
                        len(pos), 7, 1, 1).expand(-1, -1, 9, 9)
                    color_obs[:, :7] = torch.gather(
                        obs[:, :7], 1, gather_planes)
                    # Channels 8:11 encode next-ball color / 7 at fixed
                    # positions.  Relabel the scalar colors in place.
                    old_colors = torch.round(
                        obs[:, 8:11] * 7.0).long().reshape(len(pos), -1)
                    new_colors = torch.gather(
                        color_map, 1, old_colors).reshape(
                            len(pos), 3, 9, 9)
                    color_obs[:, 8:11] = new_colors.float() / 7.0
                    logits = model(color_obs.to(dtype))
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    action = logits.float().masked_fill(
                        ~legal, float('-inf')).argmax(1)
                    color_actions.append(action)
                    color_probs.append(torch.softmax(
                        logits.float().masked_fill(
                            ~legal, float('-inf')), dim=1))
                    color_agree[ci] += int((action == reference[:, 0]).sum())
                    color_teacher_matches[ci] += int(
                        (action == recorded).sum())
                stacked_colors = torch.stack(color_actions, dim=1)
                all_colors_consistent += int(
                    (stacked_colors == reference).all(1).sum())
                color_mean = torch.stack(color_probs, dim=0).mean(0)
                color_action = color_mean.argmax(1)
                color_ensemble_matches += int(
                    (color_action == recorded).sum())
                color_ensemble_reference_agree += int(
                    (color_action == reference[:, 0]).sum())
                mean_h = -(color_mean
                           * color_mean.clamp_min(1e-30).log()).sum(1)
                view_h = torch.stack([
                    -(p * p.clamp_min(1e-30).log()).sum(1)
                    for p in color_probs], dim=0).mean(0)
                color_js_sum += float((mean_h - view_h).sum())
                del color_probs, color_mean
            seen += len(pos)
            done = min(start + args.batch_size, n)
            if done % (args.batch_size * 20) == 0 or done == n:
                print(f'  {os.path.basename(path)}: {done:,}/{n:,}',
                      flush=True)

        print(f'\n[{os.path.basename(path)}] n={seen:,}', flush=True)
        print('  agreement with orientation-0 action: ' + ' '.join(
            f't{t}={100 * agree[t] / seen:.2f}%'
            for t in range(8)), flush=True)
        print(f'  nonidentity mean={100 * agree[1:].mean() / seen:.2f}% '
              f'all-eight={100 * all_consistent / seen:.2f}%', flush=True)
        print('  recorded-action match after mapping: ' + ' '.join(
            f't{t}={100 * teacher_matches[t] / seen:.2f}%'
            for t in range(8)), flush=True)
        print(f'  D4 probability ensemble: recorded-match='
              f'{100 * ensemble_matches / seen:.2f}% '
              f'orientation0-agree={100 * ensemble_reference_agree / seen:.2f}% '
              f'JS={d4_js_sum / seen:.6f}', flush=True)
        if args.color_permutations:
            print('  color-permutation agreement: ' + ' '.join(
                f'c{ci + 1}={100 * color_agree[ci] / seen:.2f}%'
                for ci in range(args.color_permutations)), flush=True)
            print(f'  color mean={100 * color_agree.mean() / seen:.2f}% '
                  f'all={100 * all_colors_consistent / seen:.2f}%',
                  flush=True)
            print('  recorded-action match after color relabeling: ' + ' '.join(
                f'c{ci + 1}={100 * color_teacher_matches[ci] / seen:.2f}%'
                for ci in range(args.color_permutations)), flush=True)
            print(f'  color probability ensemble: recorded-match='
                  f'{100 * color_ensemble_matches / seen:.2f}% '
                  f'orientation0-agree='
                  f'{100 * color_ensemble_reference_agree / seen:.2f}% '
                  f'JS={color_js_sum / seen:.6f}', flush=True)
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()


if __name__ == '__main__':
    main()
