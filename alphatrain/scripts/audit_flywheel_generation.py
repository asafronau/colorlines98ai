"""Validate resumable flywheel JSON streams before building a tensor.

The audit is intentionally streaming: a 3--6M-state pilot is large enough that
loading all root diagnostics into Python lists is wasteful. Exact counts are
kept for every row and quantiles are estimated from a deterministic reservoir.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from alphatrain.scripts.build_flywheel_corpus import (
    _atomic_json, flat_move, rows_key, source_files, validate_game,
)


class Reservoir:
    def __init__(self, size, seed):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.rows = []

    def add(self, row):
        self.seen += 1
        if len(self.rows) < self.size:
            self.rows.append(row)
            return
        slot = int(self.rng.integers(self.seen))
        if slot < self.size:
            self.rows[slot] = row


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        'mean': float(values.mean()),
        'p10': float(np.percentile(values, 10)),
        'p50': float(np.percentile(values, 50)),
        'p90': float(np.percentile(values, 90)),
    }


def exact_root_summary(counter):
    rows = int(counter['rows'])
    prior_rows = int(counter['prior_rows'])
    q_rows = int(counter['q_rows'])
    return {
        'rows': rows,
        'teacher_prior_top1_fraction': (
            counter['teacher_prior_top1'] / prior_rows
            if prior_rows else None),
        'teacher_q_top1_fraction': (
            counter['teacher_q_top1'] / q_rows if q_rows else None),
        'teacher_prior_rank': {
            'rank1': int(counter['prior_rank_1']),
            'rank2': int(counter['prior_rank_2']),
            'rank3': int(counter['prior_rank_3']),
            'rank4plus': int(counter['prior_rank_4plus']),
        },
    }


def source_audit(source, source_id, sample_rows, require_complete):
    files = source_files(source)
    expected_files = source.get('expected_files')
    marker_path = Path(source['path']) / 'generation_complete.json'
    complete = json.loads(marker_path.read_text()) if marker_path.exists() else None
    errors = []
    if expected_files is not None and len(files) != int(expected_files):
        errors.append(f'files={len(files)} expected={int(expected_files)}')
    if source.get('require_complete_marker') and complete is None:
        errors.append('generation_complete.json missing')

    game_lengths, scores, game_turns = [], [], []
    group_labels = defaultdict(set)
    exact = Counter()
    root_strata = defaultdict(Counter)
    turn_bands = Counter()
    reservoir = Reservoir(sample_rows, 20260809 + source_id)

    for fi, path in enumerate(files):
        try:
            with open(path) as handle:
                game = json.load(handle)
            rows = validate_game(source, game, path)
        except Exception as exc:  # report all corruptions before failing
            errors.append(f'{path}: {exc}')
            continue
        group_seed = int(game.get('original_seed', game.get('seed', -1)))
        group_labels[group_seed].add(str(game.get('label', 'game')))
        game_lengths.append(len(rows))
        scores.append(int(game.get('score', -1)))
        game_turns.append(int(game.get('turns', len(rows))))
        exact['games'] += 1
        exact['game_capped'] += int(bool(game.get('capped', False)))
        clean_sims = int(game.get('clean_label_sims', 0))
        noise = float(game.get('behavior_dirichlet_weight', np.nan))
        temp_moves = int(game.get('temperature_moves', 0))
        behavior_clean = np.isfinite(noise) and noise == 0 and temp_moves == 0
        is_clean = clean_sims > 0 or behavior_clean
        game_capped = bool(game.get('capped', False))

        for local_turn, row in enumerate(rows):
            exact['rows'] += 1
            exact['clean_rows'] += int(is_clean)
            band = min(local_turn // 100, 9) * 100
            turn_bands[f'{band:03d}-{band + 99:03d}'] += 1
            moves = row.get('cand_moves', [])
            visits = np.asarray(row.get('cand_visits', []), dtype=np.float64)
            if len(moves) != len(visits) or not len(moves) or visits.sum() <= 0:
                exact['invalid_target_rows'] += 1
                continue
            behavior = flat_move(row['chosen_move'])
            teacher = flat_move(row.get('teacher_move', moves[0]))
            exact['behavior_teacher_agree'] += int(behavior == teacher)
            required = ('cand_prior', 'cand_q', 'root_value', 'q_min', 'q_max')
            full = all(key in row for key in required)
            exact['full_record_rows'] += int(full)
            probs = visits / visits.sum()
            order = np.argsort(-visits, kind='stable')
            visit_margin = (probs[order[0]] - probs[order[1]]
                            if len(order) > 1 else probs[order[0]])
            item = {
                'nnz': len(moves),
                'top_share': float(probs[order[0]]),
                'visit_margin': float(visit_margin),
                'entropy': float(-(probs * np.log(np.maximum(probs, 1e-30))).sum()),
            }
            if full:
                prior = np.asarray(row['cand_prior'], dtype=np.float64)
                q = np.asarray(row['cand_q'], dtype=np.float64)
                if len(prior) == len(moves) and len(q) == len(moves):
                    teacher_slot = moves.index(teacher) if teacher in moves else 0
                    prior_order = (-prior).argsort(kind='stable')
                    prior_rank = int(
                        prior_order.tolist().index(teacher_slot) + 1)
                    item['teacher_prior_rank'] = prior_rank
                    others = np.delete(q, teacher_slot)
                    item['teacher_q_margin'] = float(
                        q[teacher_slot] - others.max()) if len(others) else 0.0
                    strata = ['all', 'capped' if game_capped else 'failed']
                    if not game_capped and local_turn >= max(len(rows) - 20, 0):
                        strata.append('failed_tail_20')
                    strata.append(f'turn_{band:03d}_{band + 99:03d}')
                    evaluated = visits > 0
                    for stratum in strata:
                        counts = root_strata[stratum]
                        counts['rows'] += 1
                        counts['prior_rows'] += 1
                        counts['teacher_prior_top1'] += int(
                            teacher_slot == int(prior_order[0]))
                        if prior_rank == 1:
                            counts['prior_rank_1'] += 1
                        elif prior_rank == 2:
                            counts['prior_rank_2'] += 1
                        elif prior_rank == 3:
                            counts['prior_rank_3'] += 1
                        else:
                            counts['prior_rank_4plus'] += 1
                        if evaluated.any() and evaluated[teacher_slot]:
                            evaluated_slots = np.flatnonzero(evaluated)
                            q_top = evaluated_slots[
                                np.argmax(q[evaluated_slots])]
                            counts['q_rows'] += 1
                            counts['teacher_q_top1'] += int(
                                teacher_slot == int(q_top))
            reservoir.add(item)
        if (fi + 1) % 500 == 0:
            print(f'  {source["name"]}: {fi+1:,}/{len(files):,} games; '
                  f'{exact["rows"]:,} rows', flush=True)

    # Crisis generation represents a completed probe either by both replay
    # labels or by an entry in capped_probes.log.
    capped_probe_path = Path(source['path']) / 'capped_probes.log'
    capped_probes = set()
    if capped_probe_path.exists():
        for line in capped_probe_path.read_text().splitlines():
            if line.strip():
                capped_probes.add(int(line))
    seed_start, seed_end = source.get('seed_start'), source.get('seed_end')
    incomplete_seeds = []
    if seed_start is not None and seed_end is not None:
        for seed in range(int(seed_start), int(seed_end)):
            labels = group_labels.get(seed, set())
            if capped_probe_path.exists():
                done = seed in capped_probes or labels >= {'recovery', 'prevention'}
            else:
                done = bool(labels)
            if not done:
                incomplete_seeds.append(seed)
    if incomplete_seeds:
        errors.append(
            f'{len(incomplete_seeds)} incomplete seeds; first='
            f'{incomplete_seeds[:10]}')

    sampled = reservoir.rows
    sampled_keys = sorted({key for row in sampled for key in row})
    sample_summary = {
        key: quantiles([row[key] for row in sampled if key in row])
        for key in sampled_keys
    }
    n_rows = exact['rows']
    result = {
        'name': source['name'],
        'path': source['path'],
        'files': len(files),
        'rows': n_rows,
        'unique_group_seeds': len(group_labels),
        'generation_complete': complete,
        'errors': errors,
        'games': {
            'length': quantiles(game_lengths), 'score': quantiles(scores),
            'turns': quantiles(game_turns),
            'capped_fraction': exact['game_capped'] / max(exact['games'], 1),
            'capped_probes': len(capped_probes),
        },
        'rows_exact': {
            'clean_fraction': exact['clean_rows'] / max(n_rows, 1),
            'full_record_fraction': exact['full_record_rows'] / max(n_rows, 1),
            'behavior_teacher_agree': (
                exact['behavior_teacher_agree'] / max(n_rows, 1)),
            'invalid_target_rows': exact['invalid_target_rows'],
            'turn_bands': dict(turn_bands),
        },
        'root_strata_exact': {
            name: exact_root_summary(counts)
            for name, counts in sorted(root_strata.items())
        },
        'root_sample': {'n': len(sampled), **sample_summary},
    }
    if require_complete and errors:
        raise ValueError(f'{source["name"]}: ' + '; '.join(errors[:5]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument(
        '--source', action='append',
        help=('Audit only the named target source. May be repeated; source IDs '
              'remain those from the full manifest.'))
    parser.add_argument('--sample-rows', type=int, default=200_000)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    requested = set(args.source or ())
    available = {source['name'] for source in manifest['target_sources']}
    unknown = requested.difference(available)
    if unknown:
        raise ValueError(
            f'unknown target sources {sorted(unknown)}; '
            f'choices={sorted(available)}')
    results = []
    for source_id, source in enumerate(manifest['target_sources']):
        if requested and source['name'] not in requested:
            continue
        print(f'[{source["name"]}] {source["path"]}', flush=True)
        results.append(source_audit(
            source, source_id, args.sample_rows, args.require_complete))
    report = {
        'schema_version': 1,
        'manifest': args.manifest,
        'total_files': sum(row['files'] for row in results),
        'total_rows': sum(row['rows'] for row in results),
        'sources': results,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.output, report)


if __name__ == '__main__':
    main()
