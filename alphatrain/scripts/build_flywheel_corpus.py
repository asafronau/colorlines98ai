"""Build lineage-pure flywheel tensors without changing target semantics.

Unlike the historical hard-CE builders, this keeps raw MCTS visit counts as
the sparse policy target and stores the behavior move separately.  Every row
is retained.  Rows whose behavior was temperature-sampled, or terminal tails
of failed crisis replays, remain available as preservation anchors but receive
a reduced search-target weight.

The output also carries source/game/seed/turn provenance and a deterministic
group split so adjacent states from one game cannot leak across train/val.

Run from the repository root, under caffeinate for real corpora:

    python -m alphatrain.scripts.build_flywheel_corpus \
      --manifest alphatrain/flywheel/vh3_iteration_0.json --kind target
    python -m alphatrain.scripts.build_flywheel_corpus \
      --manifest alphatrain/flywheel/vh3_iteration_0.json --kind anchor
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def flat_move(move):
    if isinstance(move, (int, np.integer)):
        return int(move)
    return ((int(move['sr']) * 9 + int(move['sc'])) * 81
            + int(move['tr']) * 9 + int(move['tc']))


def checkpoint_sha256(path, chunk_size=8 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def source_files(source):
    return sorted(glob.glob(os.path.join(source['path'], 'game_seed*.json')))


def _same_value(actual, expected):
    if isinstance(expected, float):
        try:
            return bool(np.isclose(float(actual), expected, rtol=0, atol=1e-7))
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_game(source, game, path):
    """Reject mixed/incomplete generator outputs before tensor construction."""
    expected = source.get('expected_game_fields', {})
    for key, value in expected.items():
        if key not in game or not _same_value(game[key], value):
            raise ValueError(
                f'{path}: expected {key}={value!r}, got {game.get(key)!r}')
    group_seed = int(game.get('original_seed', game.get('seed', -1)))
    if 'seed_start' in source and group_seed < int(source['seed_start']):
        raise ValueError(f'{path}: seed {group_seed} below source range')
    if 'seed_end' in source and group_seed >= int(source['seed_end']):
        raise ValueError(f'{path}: seed {group_seed} above source range')
    rows = game.get(rows_key(source), [])
    if source.get('require_full_record'):
        required = {
            'cand_moves', 'cand_visits', 'cand_prior', 'cand_q',
            'root_value', 'q_min', 'q_max', 'teacher_move',
        }
        for ri, row in enumerate(rows):
            missing = required.difference(row)
            if missing:
                raise ValueError(
                    f'{path}: row {ri} missing full-record fields '
                    f'{sorted(missing)}')
    return rows


def rows_key(source):
    return 'moves' if source['format'] == 'moves' else 'states'


def count_rows(sources):
    counts = []
    total = 0
    for source in sources:
        files = source_files(source)
        n_rows = 0
        for fi, path in enumerate(files):
            with open(path) as f:
                game = json.load(f)
            n_rows += len(validate_game(source, game, path))
            if (fi + 1) % 5000 == 0:
                print(f'  count {source["name"]}: {fi+1:,}/{len(files):,} '
                      f'files, {n_rows:,} rows', flush=True)
        counts.append({'name': source['name'], 'path': source['path'],
                       'files': len(files), 'rows': n_rows})
        expected_files = source.get('expected_files')
        if expected_files is not None and len(files) != int(expected_files):
            raise ValueError(
                f'{source["name"]}: expected {int(expected_files):,} files, '
                f'found {len(files):,}')
        if source.get('require_complete_marker'):
            marker = Path(source['path']) / 'generation_complete.json'
            if not marker.exists():
                raise ValueError(f'{source["name"]}: missing {marker}')
            complete = json.loads(marker.read_text())
            expected_run = source.get(
                'expected_game_fields', {}).get('run_id')
            if (expected_run is not None
                    and complete.get('run_id') != expected_run):
                raise ValueError(
                    f'{marker}: run_id {complete.get("run_id")!r} != '
                    f'{expected_run!r}')
            for key, value in source.get(
                    'expected_complete_fields', {}).items():
                if key not in complete or not _same_value(
                        complete[key], value):
                    raise ValueError(
                        f'{marker}: expected {key}={value!r}, '
                        f'got {complete.get(key)!r}')
        total += n_rows
        print(f'{source["name"]}: {len(files):,} files, {n_rows:,} rows',
              flush=True)
    return total, counts


def _array_layout(n, k):
    return {
        'boards': ((n, 9, 9), np.int8, 0),
        'next_pos': ((n, 3, 2), np.int8, 0),
        'next_col': ((n, 3), np.int8, 0),
        'n_next': ((n,), np.int8, 0),
        # Action ids fit in int16 (0..6560).  Compact storage matters once
        # thousands of independent capped search games contribute tens of
        # millions of rows; indices become int64 only for batch-local scatter.
        'pol_indices': ((n, k), np.int16, 0),
        'pol_values': ((n, k), np.float16, 0),
        'pol_nnz': ((n,), np.int8, 0),
        'behavior_move': ((n,), np.int16, -1),
        'teacher_move': ((n,), np.int16, -1),
        # Exact legal policy argmax from the actor forward which initialized
        # this search root.  New full records preserve all 30 root priors, and
        # LegalPriors selected those actions from the complete legal support,
        # so their prior argmax is the behavior-time base action.  Keeping it
        # avoids reclassifying rare edits with a different MPS batch shape.
        'base_move': ((n,), np.int16, -1),
        # Preserve the root-search evidence instead of making a later target
        # recipe guess from normalized/truncated visits.  Missing fields stay
        # NaN and full_search_record=0 for honest historical/slim provenance.
        'cand_visit': ((n, k), np.int32, 0),
        'cand_prior': ((n, k), np.float16, np.nan),
        'cand_q': ((n, k), np.float16, np.nan),
        'root_value': ((n,), np.float16, np.nan),
        'q_min': ((n,), np.float16, np.nan),
        'q_max': ((n,), np.float16, np.nan),
        'full_search_record': ((n,), np.uint8, 0),
        'target_weight': ((n,), np.float32, 0),
        'source_id': ((n,), np.int16, 0),
        'game_id': ((n,), np.int32, 0),
        'group_seed': ((n,), np.int64, 0),
        'turn': ((n,), np.int32, 0),
        'split': ((n,), np.uint8, 0),
        'game_capped': ((n,), np.int8, -1),
        'game_score': ((n,), np.int32, -1),
        'game_turns': ((n,), np.int32, -1),
        'search_sims': ((n,), np.int32, 0),
        'search_q_weight': ((n,), np.float32, np.nan),
        'search_top_k': ((n,), np.int16, 0),
        'clean_label': ((n,), np.uint8, 0),
        'behavior_noise_weight': ((n,), np.float32, np.nan),
        # 0=unknown, 1=selfplay, 2=prevention, 3=recovery, 4=greedy.
        'trajectory_kind': ((n,), np.int8, 0),
    }


def allocate(n, k, memmap_dir=None, resume=False):
    arrays = {}
    for name, (shape, dtype, fill) in _array_layout(n, k).items():
        if memmap_dir is None:
            value = np.empty(shape, dtype=dtype)
            value.fill(fill)
        else:
            path = Path(memmap_dir) / f'{name}.npy'
            mode = 'r+' if resume else 'w+'
            value = np.lib.format.open_memmap(
                path, mode=mode, dtype=dtype, shape=shape)
            if not resume:
                value.fill(fill)
        arrays[name] = value
    return arrays


def _atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def _flush(arrays):
    for value in arrays.values():
        if isinstance(value, np.memmap):
            value.flush()


def fill_position(arrays, out_i, row, source, source_id, game_id,
                  group_seed, turn, local_turn, game, kind, k):
    arrays['boards'][out_i] = np.asarray(row['board'], dtype=np.int8)
    balls = row.get('next_balls', [])[:int(row.get('num_next', 3))]
    for j, ball in enumerate(balls[:3]):
        arrays['next_pos'][out_i, j] = (int(ball['row']), int(ball['col']))
        arrays['next_col'][out_i, j] = int(ball['color'])
    arrays['n_next'][out_i] = min(len(balls), 3)
    arrays['source_id'][out_i] = source_id
    arrays['game_id'][out_i] = game_id
    arrays['group_seed'][out_i] = group_seed
    arrays['turn'][out_i] = turn
    if 'capped' in game:
        arrays['game_capped'][out_i] = int(bool(game['capped']))
    elif kind == 'anchor' and 'died' in game:
        # Native greedy recordings use final_* plus died, whereas native MCTS
        # files use score/turns/capped.  Preserve equivalent diagnostics for
        # both formats instead of silently writing -1 for every new anchor.
        arrays['game_capped'][out_i] = int(not bool(game['died']))
    arrays['game_score'][out_i] = int(
        game.get('score', game.get('final_score', -1)))
    arrays['game_turns'][out_i] = int(
        game.get('turns', game.get('final_turns', -1)))
    # New self-play files always declare clean_label_sims, including zero.
    # Zero means candidates came from the behavior tree, not that no search
    # was run, so only prefer the clean budget when it is actually enabled.
    clean_sims = int(game.get('clean_label_sims', 0))
    arrays['search_sims'][out_i] = (
        clean_sims if clean_sims > 0 else int(game.get(
            'behavior_sims', game.get(
                'replay_sims', game.get(
                    'simulations', source.get('search_sims', 0))))))
    arrays['search_q_weight'][out_i] = float(
        game.get('q_weight', source.get('q_weight', np.nan)))
    arrays['search_top_k'][out_i] = int(
        game.get('top_k', source.get('top_k', 0)))
    behavior_noise = float(game.get(
        'behavior_dirichlet_weight',
        source.get('dirichlet_weight', np.nan)))
    temperature_moves = int(game.get(
        'temperature_moves', source.get('temperature_moves', 0)))
    # A separate clean tree is not required when the behavior tree itself is
    # explicitly noise-free and always takes its visit winner.  This mode is
    # useful for large banks of capped exploit trajectories: it gives a
    # faithful clean label at half the cost of behavior+reanalysis per move.
    behavior_is_clean = (
        np.isfinite(behavior_noise) and behavior_noise == 0.0
        and temperature_moves == 0)
    arrays['clean_label'][out_i] = int(clean_sims > 0 or behavior_is_clean)
    arrays['behavior_noise_weight'][out_i] = behavior_noise
    label = str(game.get('label', '')).lower()
    if label == 'prevention':
        arrays['trajectory_kind'][out_i] = 2
    elif label == 'recovery':
        arrays['trajectory_kind'][out_i] = 3
    elif kind == 'anchor':
        arrays['trajectory_kind'][out_i] = 4
    else:
        arrays['trajectory_kind'][out_i] = 1
    # Five percent of source-game groups form validation.  The same original
    # seed therefore keeps prevention/recovery siblings together.
    arrays['split'][out_i] = 1 if (int(group_seed) % 20 == 0) else 0

    if kind == 'anchor':
        action = flat_move(row['move'])
        arrays['behavior_move'][out_i] = action
        arrays['pol_indices'][out_i, 0] = action
        arrays['pol_values'][out_i, 0] = 1.0
        arrays['pol_nnz'][out_i] = 1
        return

    action = flat_move(row['chosen_move'])
    arrays['behavior_move'][out_i] = action
    moves = np.asarray(row.get('cand_moves', [])[:k], dtype=np.int64)
    raw_visits = np.asarray(
        row.get('cand_visits', [])[:len(moves)], dtype=np.int64)
    if len(raw_visits) != len(moves):
        raw_visits = np.zeros(len(moves), dtype=np.int64)
    raw_prior = np.asarray(
        row.get('cand_prior', [])[:len(moves)], dtype=np.float64)
    raw_q = np.asarray(
        row.get('cand_q', [])[:len(moves)], dtype=np.float64)
    has_full = (len(moves) > 0 and len(raw_prior) == len(moves)
                and len(raw_q) == len(moves)
                and all(name in row
                        for name in ('root_value', 'q_min', 'q_max')))
    if has_full:
        arrays['base_move'][out_i] = int(moves[np.argmax(raw_prior)])
    visits = raw_visits.astype(np.float64)
    positive = visits > 0
    moves, visits = moves[positive], visits[positive]
    raw_visits = raw_visits[positive]
    nnz = len(moves)
    if nnz:
        arrays['cand_visit'][out_i, :nnz] = raw_visits.astype(np.int32)
        if has_full:
            arrays['cand_prior'][out_i, :nnz] = raw_prior[positive]
            arrays['cand_q'][out_i, :nnz] = raw_q[positive]
            arrays['root_value'][out_i] = float(row['root_value'])
            arrays['q_min'][out_i] = float(row['q_min'])
            arrays['q_max'][out_i] = float(row['q_max'])
            arrays['full_search_record'][out_i] = 1
        visits /= visits.sum()
        arrays['pol_indices'][out_i, :nnz] = moves
        arrays['pol_values'][out_i, :nnz] = visits.astype(np.float16)
        arrays['pol_nnz'][out_i] = nnz
    teacher = row.get('teacher_move', int(moves[0]) if nnz else action)
    arrays['teacher_move'][out_i] = flat_move(teacher)

    weight = 1.0 if nnz else 0.0
    temp_moves = int(game.get(
        'temperature_moves', source.get('temperature_moves', 0)))
    # Exploratory behavior is still a valid training state when an independent
    # noise-free tree supplies the label.  Only the historical/noisy behavior
    # tree itself is disabled during temperature-sampled turns.
    if local_turn < temp_moves and clean_sims <= 0:
        weight = 0.0
    if (source['format'] == 'moves' and not game.get('capped', True)
            and local_turn >= len(game.get('moves', [])) - 20):
        weight *= float(source.get('failed_tail_weight', 1.0))
    arrays['target_weight'][out_i] = weight


def build(manifest_path, kind, output=None, top_k=30, resume_dir=None,
          checkpoint_every_files=25):
    if checkpoint_every_files <= 0:
        raise ValueError('checkpoint_every_files must be positive')
    with open(manifest_path) as f:
        manifest = json.load(f)
    sources = manifest[f'{kind}_sources']
    # Keep the historical small128 manifests valid, but make the safety gate
    # describe the model *family* rather than one exact architecture.  New
    # compact actors (for example 18 blocks x 96 channels) must explicitly opt
    # into the small-policy family and declare a generator prefix.  That prefix
    # is then enforced for every source, preventing accidental mixing with
    # games from a large teacher or a previous actor lineage.
    lineage = str(manifest.get('lineage', ''))
    if lineage == 'small128':
        generator_prefix = str(
            manifest.get('source_generator_prefix', 'small128_'))
    elif manifest.get('lineage_family') == 'small_policy':
        generator_prefix = str(
            manifest.get('source_generator_prefix', ''))
        if not generator_prefix:
            raise ValueError(
                'small-policy manifest requires source_generator_prefix')
    else:
        raise ValueError(
            'primary flywheel builder requires legacy lineage=small128 or '
            'lineage_family=small_policy')
    for source in sources:
        if not str(source.get('generator', '')).startswith(generator_prefix):
            raise ValueError(
                f'source generator does not match lineage prefix '
                f'{generator_prefix!r}: {source}')

    base_path = manifest['base_checkpoint']
    base_sha256 = checkpoint_sha256(base_path)
    declared_base_sha256 = manifest.get(
        'generation_provenance', {}).get('policy_checkpoint_sha256')
    if (declared_base_sha256 is not None
            and base_sha256 != declared_base_sha256):
        raise ValueError(
            f'{base_path}: sha256 {base_sha256} != manifest '
            f'{declared_base_sha256}')

    n, source_counts = count_rows(sources)
    k = top_k if kind == 'target' else 1
    output = output or manifest[f'{kind}_output']
    manifest_hash = checkpoint_sha256(manifest_path)
    files_by_source = [source_files(source) for source in sources]
    offsets = np.cumsum([0] + [len(files) for files in files_by_source])
    jobs = [(source_id, source, path)
            for source_id, (source, files) in enumerate(
                zip(sources, files_by_source))
            for path in files]
    inventory = hashlib.sha256()
    for _, _, path in jobs:
        stat = os.stat(path)
        inventory.update(
            f'{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n'.encode())
    inventory_hash = inventory.hexdigest()

    progress_path = None
    start_job = out_i = 0
    if resume_dir:
        resume_dir = Path(resume_dir)
        resume_dir.mkdir(parents=True, exist_ok=True)
        progress_path = resume_dir / 'progress.json'
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            expected = {
                'schema_version': 1, 'manifest_sha256': manifest_hash,
                'kind': kind, 'top_k': top_k, 'rows': n,
                'files': len(jobs), 'output': str(output),
                'input_inventory_sha256': inventory_hash,
            }
            for key, value in expected.items():
                if progress.get(key) != value:
                    raise ValueError(
                        f'{progress_path}: {key}={progress.get(key)!r}, '
                        f'expected {value!r}')
            start_job = int(progress.get('next_file', 0))
            out_i = int(progress.get('rows_written', 0))
            arrays = allocate(n, k, resume_dir, resume=True)
            print(f'resume build: file {start_job:,}/{len(jobs):,}; '
                  f'rows {out_i:,}/{n:,}', flush=True)
        else:
            arrays = allocate(n, k, resume_dir, resume=False)
            _flush(arrays)
            progress = {
                'schema_version': 1, 'manifest_sha256': manifest_hash,
                'kind': kind, 'top_k': top_k, 'rows': n,
                'files': len(jobs), 'output': str(output),
                'input_inventory_sha256': inventory_hash,
                'next_file': 0, 'rows_written': 0,
                'started_unix': time.time(), 'complete': False,
            }
            _atomic_json(progress_path, progress)
    else:
        arrays = allocate(n, k)

    for job_i, (source_id, source, path) in enumerate(jobs):
        if job_i < start_job:
            continue
        key = rows_key(source)
        fi = job_i - int(offsets[source_id])
        files_in_source = len(files_by_source[source_id])
        game_id = job_i
        with open(path) as f:
            game = json.load(f)
        validate_game(source, game, path)
        group_seed = int(game.get('original_seed', game['seed']))
        base_turn = int(game.get('replay_from_turn', 0))
        for local_turn, row in enumerate(game.get(key, [])):
            turn = (int(row.get('turn', local_turn)) if key == 'states'
                    else base_turn + local_turn)
            fill_position(arrays, out_i, row, source, source_id, game_id,
                          group_seed, turn, local_turn, game, kind, k)
            out_i += 1
        if progress_path and (
                (job_i + 1) % checkpoint_every_files == 0
                or job_i + 1 == len(jobs)):
            _flush(arrays)
            progress.update({
                'next_file': job_i + 1, 'rows_written': out_i,
                'updated_unix': time.time(),
            })
            _atomic_json(progress_path, progress)
        if (fi + 1) % 1000 == 0 or job_i + 1 == len(jobs):
            print(f'  fill {source["name"]}: {fi+1:,}/{files_in_source:,} '
                  f'files; total {out_i:,}/{n:,}', flush=True)
    assert out_i == n

    metadata = {
        'schema_version': 4,
        'iteration': manifest['iteration'],
        'lineage': manifest['lineage'],
        'kind': kind,
        'base_checkpoint': base_path,
        'base_checkpoint_sha256': base_sha256,
        'label_protocol': manifest.get('label_protocol', 'unknown'),
        'source_counts': source_counts,
        'source_names': [s['name'] for s in sources],
        'manifest_path': str(manifest_path),
        'manifest_sha256': manifest_hash,
        'input_inventory_sha256': inventory_hash,
        'generation_provenance': manifest.get('generation_provenance', {}),
        'source_definitions': sources,
        'target_semantics': ('raw_positive_MCTS_visits_plus_teacher_move'
                             if kind == 'target'
                             else 'behavior_move_for_legacy_control_only'),
        'search_detail_semantics': (
            'raw cand_visit; cand_prior stores clean log-prior; cand_q and '
            'root values retained when full_search_record=1; base_move is '
            'the exact legal clean-prior argmax from that actor root'),
        'sparse_top_k': k,
        'split_semantics': 'group_seed_mod_20; 1=validation',
        'trajectory_kind_names': {
            0: 'unknown', 1: 'selfplay', 2: 'prevention',
            3: 'recovery', 4: 'greedy'},
    }
    tensor = {name: torch.from_numpy(value) for name, value in arrays.items()}
    tensor.update({'num_channels': 18, 'max_score': 0.0,
                   'value_mode': 'policy_slim', 'metadata': metadata})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tmp = output + '.tmp'
    torch.save(tensor, tmp)
    os.replace(tmp, output)
    if progress_path:
        progress.update({
            'next_file': len(jobs), 'rows_written': out_i,
            'updated_unix': time.time(), 'complete': True,
            'output_size': os.path.getsize(output),
        })
        _atomic_json(progress_path, progress)
    nonzero = int((arrays['target_weight'] > 0).sum())
    print(f'{output}: {n:,} rows; target-weight>0 {nonzero:,}; '
          f'val groups/rows={int((arrays["split"] == 1).sum()):,}', flush=True)
    return tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--kind', choices=['target', 'anchor'], required=True)
    p.add_argument('--output')
    p.add_argument('--top-k', type=int, default=30,
                   help='Keep the complete default MCTS root support.')
    p.add_argument('--resume-dir',
                   help='Memmapped build state. Re-running resumes at the '
                        'last flushed source file.')
    p.add_argument('--checkpoint-every-files', type=int, default=25)
    args = p.parse_args()
    build(args.manifest, args.kind, args.output, args.top_k,
          args.resume_dir, args.checkpoint_every_files)


if __name__ == '__main__':
    main()
