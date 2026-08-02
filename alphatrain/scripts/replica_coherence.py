"""Shuffle-replica coherence test (review #6): train K replicas of the
advantage fine-tune varying only the shuffle seed; report pairwise cosines of
the task vectors in parameter space. Near-orthogonal replicas = optimizer/data
variance exceeds the signal (averaging would erase, not denoise).

    python -m alphatrain.scripts.replica_coherence \
        --corpus alphatrain/data/advfilt.pt \
        --base alphatrain/data/small128_vh1.pt --k 8
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch


def flat_delta(ck_path, base_sd):
    sd = torch.load(ck_path, map_location='cpu', weights_only=False)['model']
    return torch.cat([(sd[k].float() - base_sd[k].float()).flatten()
                      for k in sorted(sd) if 'running_' not in k
                      and 'num_batches' not in k]).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--anchor-games-dir', default='data/dagger_games_v1')
    p.add_argument('--out-dir', default='checkpoints/replicas')
    a = p.parse_args()

    base_sd = torch.load(a.base, map_location='cpu', weights_only=False)
    base_sd = base_sd['model'] if 'model' in base_sd else base_sd
    deltas = []
    for s in range(a.k):
        sdir = f'{a.out_dir}/s{s}'
        ck = f'{sdir}/ft_epoch_{a.epochs}.pt'
        if not os.path.exists(ck):
            print(f'--- replica seed {s} ---', flush=True)
            env = dict(os.environ, PYTHONPATH='.')
            subprocess.run(
                [sys.executable, 'scripts/train_crisis_ft.py',
                 '--corpus', a.corpus, '--base', a.base,
                 '--loss', 'soft', '--epochs', str(a.epochs),
                 '--lr', '1e-4', '--batch', '512',
                 '--kl-anchor-weight', '3.0',
                 '--anchor-games-dir', a.anchor_games_dir,
                 '--holdout-frac', '0.15', '--shuffle-seed', str(s),
                 '--save-every', str(a.epochs), '--save-dir', sdir],
                check=True, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        deltas.append(flat_delta(ck, base_sd))
    D = np.stack(deltas)
    norms = np.linalg.norm(D, axis=1)
    cos = (D @ D.T) / np.outer(norms, norms)
    off = cos[np.triu_indices(a.k, 1)]
    print(f'\n{a.k} replicas of {a.corpus}:')
    print(f'  |delta| mean {norms.mean():.4f} (cv {norms.std()/norms.mean():.2f})')
    print(f'  pairwise cos: mean {off.mean():+.3f}  min {off.min():+.3f}  '
          f'max {off.max():+.3f}')
    print('  COHERENT (avg would denoise)' if off.mean() > 0.5 else
          '  WEAK/ORTHOGONAL (avg would erase)' if off.mean() < 0.2 else
          '  INTERMEDIATE')


if __name__ == '__main__':
    main()
