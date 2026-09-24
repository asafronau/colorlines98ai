"""Compute a FULL-LEGAL base-policy disagreement sidecar for a corpus.

mask[i] = 1 where the corpus target argmax != the base policy's legal argmax
over ALL legal moves at the requested precision.  The inference backend and
batch shape are recorded because MPS outputs can vary with both; call this a
deployment proxy unless they exactly match the evaluator.

    python -m alphatrain.scripts.add_fulllegal_mask \
        --tensor alphatrain/data/vh2c_crisis.pt \
        --base alphatrain/data/small128_vh1.pt \
        --output alphatrain/data/vh2c_crisis_base_policy.npz

Use ``--in-place`` only for legacy trainers which require ``disagree_mask``
inside the tensor.  The sidecar default keeps expensive source tensors
immutable and also records the actual base/target moves and target confidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.mcts import _legal_argmax_batch
from alphatrain.evaluate import load_model


def sha256(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    tmp = str(path) + '.tmp'
    Path(tmp).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', required=True)
    p.add_argument('--base', default='alphatrain/data/small128_vh1.pt')
    p.add_argument('--device', default='mps')
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--fp32', action='store_true',
                   help='Use fp32 even on MPS/CUDA (default there is fp16).')
    p.add_argument('--output', default=None,
                   help='Sidecar .npz path (default: <tensor>_base_policy.npz).')
    p.add_argument('--in-place', action='store_true',
                   help='Also write disagree_mask back into --tensor (legacy).')
    p.add_argument('--checkpoint-batches', type=int, default=25,
                   help='Flush resumable base-action state this often.')
    a = p.parse_args()
    if a.checkpoint_batches <= 0:
        raise ValueError('--checkpoint-batches must be positive')
    dev = torch.device(a.device)

    d = torch.load(a.tensor, map_location='cpu', weights_only=False, mmap=True)
    n = d['boards'].shape[0]
    visit_argmax = d['pol_indices'][
        torch.arange(n), d['pol_values'].argmax(1)].numpy()
    tgt = (d['teacher_move'].numpy()
           if 'teacher_move' in d else visit_argmax).copy()
    missing_teacher = tgt < 0
    tgt[missing_teacher] = visit_argmax[missing_teacher]
    target_top_share = d['pol_values'].max(1).values.numpy()
    recorded_base = (d['base_move'].numpy().astype(np.int16, copy=False)
                     if 'base_move' in d else None)
    ds = TensorDatasetGPU(a.tensor, augment=False, color_augment=False,
                          augment_factor=1, device=a.device)
    use_fp16 = dev.type in ('mps', 'cuda') and not a.fp32
    net, _ = load_model(a.base, dev, fp16=use_fp16)
    infer_dtype = torch.float16 if use_fp16 else torch.float32
    out = a.output or a.tensor.removesuffix('.pt') + '_base_policy.npz'
    partial = out + '.base_argmax.partial.npy'
    progress_path = out + '.progress.json'
    tensor_stat = os.stat(a.tensor)
    identity = {
        'schema_version': 2,
        'tensor': a.tensor,
        'tensor_size': tensor_stat.st_size,
        'tensor_mtime_ns': tensor_stat.st_mtime_ns,
        'tensor_manifest_sha256': (d.get('metadata', {}) or {}).get(
            'manifest_sha256'),
        'tensor_inventory_sha256': (d.get('metadata', {}) or {}).get(
            'input_inventory_sha256'),
        'base': a.base,
        'base_sha256': sha256(a.base),
        'rows': n,
        'batch_size': a.batch_size,
        'device': a.device,
        'precision': 'fp16' if use_fp16 else 'fp32',
    }
    start = 0
    if Path(progress_path).exists():
        progress = json.loads(Path(progress_path).read_text())
        for key, value in identity.items():
            if progress.get(key) != value:
                raise ValueError(
                    f'{progress_path}: {key}={progress.get(key)!r}, '
                    f'expected {value!r}')
        start = int(progress.get('next_row', 0))
        varg = np.lib.format.open_memmap(partial, mode='r+')
        if varg.shape != (n,) or varg.dtype != np.int16:
            raise ValueError(f'{partial}: wrong shape/dtype')
        print(f'resume base-policy annotation at {start:,}/{n:,}', flush=True)
    else:
        varg = np.lib.format.open_memmap(
            partial, mode='w+', dtype=np.int16, shape=(n,))
        varg.fill(-1)
        varg.flush()
        progress = {**identity, 'next_row': 0, 'complete': False}
        atomic_json(progress_path, progress)
    for batch_i, s in enumerate(range(start, n, a.batch_size), 1):
        e = min(s + a.batch_size, n)
        obs = ds._build_obs_core(ds.boards[s:e], next_pos=ds.next_pos[s:e],
                                 next_col=ds.next_col[s:e], n_next=ds.n_next[s:e])
        with torch.no_grad():
            lg = net(obs.to(infer_dtype)).float().cpu().numpy()
        bd = ds.boards[s:e].cpu().numpy().astype(np.int8)
        varg[s:e] = _legal_argmax_batch(bd, lg).astype(np.int16)
        if batch_i % a.checkpoint_batches == 0 or e == n:
            varg.flush()
            progress.update({'next_row': e})
            atomic_json(progress_path, progress)
        if batch_i % 10 == 0 or e == n:
            print(f'  {e:,}/{n:,}', flush=True)
    mask_np = (tgt != varg).astype(np.int8)
    recorded_valid = ((recorded_base >= 0) if recorded_base is not None
                      else np.zeros(n, dtype=bool))
    recorded_or_recomputed = np.asarray(varg).copy()
    if recorded_base is not None:
        recorded_or_recomputed[recorded_valid] = recorded_base[recorded_valid]
    recorded_mask_np = (tgt != recorded_or_recomputed).astype(np.int8)
    protocol = ('full-legal eager ' + ('fp16' if use_fp16 else 'fp32') +
                f' batch={a.batch_size} device={a.device} vs ' + a.base)
    metadata = {
        **identity, 'protocol': protocol,
        'teacher_move_semantics': 'teacher_move_then_visit_fallback',
        'recorded_base_semantics': (
            'base_move actor-root clean-prior legal argmax; deployment '
            'recompute fallback where unavailable'),
        'recorded_base_valid_rows': int(recorded_valid.sum()),
    }
    tmp = out + '.tmp'
    with open(tmp, 'wb') as handle:
        np.savez(handle, base_argmax=np.asarray(varg), target_argmax=tgt,
                 target_top_share=target_top_share, disagree=mask_np,
                 recorded_base_argmax=(
                     recorded_base if recorded_base is not None
                     else np.full(n, -1, dtype=np.int16)),
                 recorded_base_valid=recorded_valid.astype(np.int8),
                 recorded_disagree=recorded_mask_np,
                 protocol=np.asarray(protocol),
                 metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    os.replace(tmp, out)
    progress.update({'next_row': n, 'complete': True,
                     'output_size': os.path.getsize(out)})
    atomic_json(progress_path, progress)
    print(f'{out}: {int(mask_np.sum()):,}/{n:,} disagreements '
          f'({100 * mask_np.mean():.1f}%)', flush=True)
    if recorded_valid.any():
        root_disagree = recorded_mask_np[recorded_valid]
        recompute_agree = np.asarray(varg)[recorded_valid] == recorded_base[
            recorded_valid]
        print('  recorded actor-root disagreements: '
              f'{int(root_disagree.sum()):,}/{int(recorded_valid.sum()):,} '
              f'({100 * root_disagree.mean():.2f}%); deployment recompute '
              f'agrees with recorded base on {100 * recompute_agree.mean():.3f}%',
              flush=True)
    if a.in_place:
        mask = torch.from_numpy(mask_np)
        d['disagree_mask'] = mask
        d['disagree_mask_protocol'] = protocol
        tensor_tmp = a.tensor + '.tmp'
        torch.save(d, tensor_tmp)
        os.replace(tensor_tmp, a.tensor)
        print(f'{a.tensor}: disagree_mask updated in-place', flush=True)


if __name__ == '__main__':
    main()
