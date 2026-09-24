"""Assemble a large, provenance-preserving state tensor for distillation.

This builder deliberately separates *state distribution* from *policy target*.
It combines slices of existing tensors with additional game JSONs, writes only
the compact state fields needed to construct observations, and stores a dummy
one-action policy.  Run ``distill_relabel`` on the result before training so
every row receives targets from one teacher with one target convention.

The tensor-slice form avoids reparsing multi-gigabyte historical JSON corpora::

    python -m alphatrain.scripts.build_master_distill_states \
      --tensor-slice v14_sp data/v14.pt 0 100000 selfplay 3f \
      --game-source v17_crisis ../data/crisis_v17 crisis 3k \
      --output data/master_states.pt

``--game-source-newest`` selects the N most recently modified game files.  It
exists for append-only directories where an older tensor already contains the
prefix; the selected manifest and its hash are recorded in metadata.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Source:
    name: str
    kind: str
    lineage: str
    mode: str
    path: str
    start: int = 0
    end: int = 0
    files: list[str] | None = None
    lengths: list[int] | None = None

    @property
    def rows(self) -> int:
        if self.mode == 'tensor':
            return self.end - self.start
        assert self.lengths is not None
        return sum(self.lengths)


def _parse_bool_kind(kind: str) -> int:
    if kind not in ('selfplay', 'crisis'):
        raise ValueError(f'kind must be selfplay or crisis, got {kind!r}')
    return int(kind == 'crisis')


def _lineage_id(lineage: str) -> int:
    if lineage not in ('3f', '3k'):
        raise ValueError(f'lineage must be 3f or 3k, got {lineage!r}')
    return int(lineage == '3k')


def _game_files(path: str, newest: int | None) -> list[str]:
    files = glob.glob(os.path.join(path, 'game_seed*.json'))
    if newest is not None:
        if newest <= 0 or newest > len(files):
            raise ValueError(
                f'newest={newest} invalid for {path} ({len(files)} files)')
        files = sorted(files, key=lambda p: (os.path.getmtime(p), p))[-newest:]
    return sorted(files)


def _count_games(source: Source) -> None:
    assert source.files is not None
    lengths = []
    for i, path in enumerate(source.files):
        with open(path) as fh:
            lengths.append(len(json.load(fh)['moves']))
        if (i + 1) % 500 == 0:
            print(f'  count {source.name}: {i + 1}/{len(source.files)} games',
                  flush=True)
    source.lengths = lengths


def _manifest_hash(files: list[str]) -> str:
    payload = '\n'.join(os.path.abspath(p) for p in files).encode()
    return hashlib.sha256(payload).hexdigest()


def _copy_tensor_source(data: dict, source: Source, out: dict,
                        dst_start: int, chunk: int) -> None:
    required = ('boards', 'next_pos', 'next_col', 'n_next',
                'pol_indices', 'pol_values')
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f'{source.path} missing {missing}')
    n = data['boards'].shape[0]
    if not (0 <= source.start < source.end <= n):
        raise ValueError(
            f'bad slice [{source.start},{source.end}) for {source.path}, n={n}')

    for rel in range(0, source.rows, chunk):
        count = min(chunk, source.rows - rel)
        src = slice(source.start + rel, source.start + rel + count)
        dst = slice(dst_start + rel, dst_start + rel + count)
        for key in ('boards', 'next_pos', 'next_col', 'n_next'):
            out[key][dst].copy_(data[key][src])
        # Placeholder only; uniform teacher relabeling overwrites this policy.
        out['pol_indices'][dst, 0].copy_(data['pol_indices'][src, 0])
        valid = data['pol_values'][src, 0] > 0
        out['pol_values'][dst, 0].copy_(valid.to(torch.float32))
        out['pol_nnz'][dst].copy_(valid.to(torch.int64))
        if 'turns_remaining' in data:
            out['turns_remaining'][dst].copy_(data['turns_remaining'][src])
        else:
            out['turns_remaining'][dst].zero_()


def _moves_to_arrays(moves: list[dict]):
    n = len(moves)
    boards = np.empty((n, 9, 9), dtype=np.int8)
    next_pos = np.zeros((n, 3, 2), dtype=np.int8)
    next_col = np.zeros((n, 3), dtype=np.int8)
    n_next = np.zeros(n, dtype=np.int8)
    actions = np.zeros(n, dtype=np.int64)
    for i, move in enumerate(moves):
        boards[i] = move['board']
        balls = move.get('next_balls', ())
        nn = min(int(move.get('num_next', len(balls))), len(balls), 3)
        n_next[i] = nn
        for j in range(nn):
            ball = balls[j]
            next_pos[i, j] = (ball['row'], ball['col'])
            next_col[i, j] = ball['color']
        chosen = move['chosen_move']
        src = chosen['sr'] * 9 + chosen['sc']
        tgt = chosen['tr'] * 9 + chosen['tc']
        actions[i] = src * 81 + tgt
    return boards, next_pos, next_col, n_next, actions


def _copy_game_source(source: Source, out: dict, dst_start: int,
                      first_game_id: int) -> int:
    assert source.files is not None and source.lengths is not None
    pos = dst_start
    game_id = first_game_id
    for i, (path, expected) in enumerate(zip(source.files, source.lengths)):
        with open(path) as fh:
            moves = json.load(fh)['moves']
        if len(moves) != expected:
            raise RuntimeError(f'{path} changed during build')
        arrays = _moves_to_arrays(moves)
        dst = slice(pos, pos + expected)
        for key, values in zip(
                ('boards', 'next_pos', 'next_col', 'n_next'), arrays[:4]):
            out[key][dst].copy_(torch.from_numpy(values))
        out['pol_indices'][dst, 0].copy_(torch.from_numpy(arrays[4]))
        out['pol_values'][dst, 0].fill_(1)
        out['pol_nnz'][dst].fill_(1)
        out['turns_remaining'][dst].copy_(
            torch.arange(expected, 0, -1, dtype=torch.int32))
        out['game_id'][dst].fill_(game_id)
        pos += expected
        game_id += 1
        if (i + 1) % 100 == 0:
            print(f'  fill {source.name}: {i + 1}/{len(source.files)} games',
                  flush=True)
    return game_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--tensor-slice', action='append', nargs=6, default=[],
                   metavar=('NAME', 'PATH', 'START', 'END', 'KIND', 'LINEAGE'))
    p.add_argument('--game-source', action='append', nargs=4, default=[],
                   metavar=('NAME', 'DIR', 'KIND', 'LINEAGE'))
    p.add_argument('--game-source-newest', action='append', nargs=5, default=[],
                   metavar=('NAME', 'DIR', 'KIND', 'LINEAGE', 'N'))
    p.add_argument('--output', required=True)
    p.add_argument('--copy-chunk', type=int, default=250_000)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()

    sources: list[Source] = []
    for name, path, start, end, kind, lineage in args.tensor_slice:
        _parse_bool_kind(kind)
        _lineage_id(lineage)
        backing = torch.load(path, map_location='cpu', weights_only=False,
                             mmap=True)
        n = int(backing['boards'].shape[0])
        del backing
        lo = int(start)
        hi = n if end in ('-1', 'end') else int(end)
        sources.append(Source(name, kind, lineage, 'tensor', path, lo, hi))

    for values, newest in (
            *((v, None) for v in args.game_source),
            *((v[:4], int(v[4])) for v in args.game_source_newest)):
        name, path, kind, lineage = values
        _parse_bool_kind(kind)
        _lineage_id(lineage)
        source = Source(name, kind, lineage, 'games', path,
                        files=_game_files(path, newest))
        if not source.files:
            raise FileNotFoundError(f'no game_seed*.json files in {path}')
        _count_games(source)
        sources.append(source)

    if not sources:
        raise SystemExit('at least one source is required')

    total = sum(source.rows for source in sources)
    print('\nComposition:', flush=True)
    for i, source in enumerate(sources):
        games = len(source.files) if source.files is not None else 'tensor'
        print(f'  {i:2d} {source.name:24s} {source.rows:>10,} rows  '
              f'{source.kind:8s} lineage={source.lineage} games={games}',
              flush=True)
    print(f'  TOTAL {total:,} rows', flush=True)
    if args.plan_only:
        return

    t0 = time.time()
    data = {
        'boards': torch.empty((total, 9, 9), dtype=torch.int8),
        'next_pos': torch.empty((total, 3, 2), dtype=torch.int8),
        'next_col': torch.empty((total, 3), dtype=torch.int8),
        'n_next': torch.empty(total, dtype=torch.int8),
        # A compact placeholder policy makes the artifact compatible with
        # TensorDatasetGPU. distill_relabel replaces it with teacher top-K.
        'pol_indices': torch.empty((total, 1), dtype=torch.int64),
        'pol_values': torch.empty((total, 1), dtype=torch.float32),
        'pol_nnz': torch.empty(total, dtype=torch.int64),
        'turns_remaining': torch.empty(total, dtype=torch.int32),
        'source_id': torch.empty(total, dtype=torch.int16),
        'lineage_id': torch.empty(total, dtype=torch.int8),
        # Existing tensor slices predate game IDs. New JSON rows retain them;
        # -1 explicitly means unavailable, never a shared pseudo-game.
        'game_id': torch.full((total,), -1, dtype=torch.int32),
        # Reuse the legacy trainer's weighting hook as declared crisis
        # provenance. The primary capacity arm uses disagree_gamma=0.
        'disagree_mask': torch.empty(total, dtype=torch.int8),
        'max_score': 0.0,
        'num_channels': 18,
        'value_mode': 'distill_states_unlabeled',
    }

    metadata_sources = []
    pos = 0
    next_game_id = 0
    for source_id, source in enumerate(sources):
        dst = slice(pos, pos + source.rows)
        data['source_id'][dst].fill_(source_id)
        data['lineage_id'][dst].fill_(_lineage_id(source.lineage))
        data['disagree_mask'][dst].fill_(_parse_bool_kind(source.kind))
        if source.mode == 'tensor':
            print(f'copying {source.name}...', flush=True)
            backing = torch.load(source.path, map_location='cpu',
                                 weights_only=False, mmap=True)
            _copy_tensor_source(backing, source, data, pos, args.copy_chunk)
            del backing
        else:
            print(f'copying {source.name} JSON games...', flush=True)
            next_game_id = _copy_game_source(
                source, data, pos, next_game_id)
        record = {
            'source_id': source_id, 'name': source.name,
            'kind': source.kind, 'lineage': source.lineage,
            'rows': source.rows, 'mode': source.mode,
            'path': os.path.abspath(source.path),
        }
        if source.mode == 'tensor':
            record['slice'] = [source.start, source.end]
        else:
            assert source.files is not None
            record['games'] = len(source.files)
            record['manifest_sha256'] = _manifest_hash(source.files)
            record['first_file'] = os.path.basename(source.files[0])
            record['last_file'] = os.path.basename(source.files[-1])
        metadata_sources.append(record)
        pos += source.rows
    assert pos == total

    data['metadata'] = {
        'builder': 'build_master_distill_states',
        'total_rows': total,
        'sources': metadata_sources,
        'policy_status': 'PLACEHOLDER: run distill_relabel before training',
        'crisis_mask_field': 'disagree_mask',
        'lineage_ids': {'3f': 0, '3k': 1},
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    tmp = args.output + '.tmp'
    print(f'saving {args.output}...', flush=True)
    torch.save(data, tmp)
    os.replace(tmp, args.output)
    print(f'done: {total:,} rows, {os.path.getsize(args.output):,} bytes, '
          f'{time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
