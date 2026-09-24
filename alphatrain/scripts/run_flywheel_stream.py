"""Run one manifest-declared native flywheel stream safely and resumably."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument(
        '--source', required=True, action='append',
        help=('Exact target_sources[].name to run. Repeat the option to run '
              'multiple streams sequentially on one MPS workload.'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest = json.loads(manifest_path.read_text())
    if len(set(args.source)) != len(args.source):
        raise ValueError(f'duplicate --source values: {args.source}')
    available = [source['name'] for source in manifest['target_sources']]
    selected = []
    for source_name in args.source:
        matches = [source for source in manifest['target_sources']
                   if source['name'] == source_name]
        if len(matches) != 1:
            raise ValueError(
                f'{source_name!r} not uniquely present; choices={available}')
        source = matches[0]
        command = source.get('native_command')
        if not command:
            raise ValueError(f'{source_name}: no native_command in manifest')
        selected.append((source, command))

    provenance = manifest['generation_provenance']
    hashes = (
        (manifest['base_checkpoint'], provenance['policy_checkpoint_sha256']),
        (manifest['base_policy_ts'], provenance['policy_ts_sha256']),
        (manifest['feature_value'], provenance['feature_value_sha256']),
    )
    for relative, expected in hashes:
        path = repo_root / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f'{path}: sha256 {actual} != manifest {expected}')

    native_cwd = repo_root / 'alphatrain' / 'inference_cpp'
    reserved_lo, reserved_hi = provenance['reserved_final_seed_range']
    for source, command in selected:
        start, end = int(source['seed_start']), int(source['seed_end'])
        if max(start, reserved_lo) < min(end, reserved_hi):
            raise ValueError(
                f'{source["name"]}: generation range intersects reserved '
                'final seeds')
        full_command = ['caffeinate', '-i', '-s', *map(str, command)]
        print(f'source: {source["name"]}', flush=True)
        print(f'cwd: {native_cwd}', flush=True)
        print('command: ' + ' '.join(full_command), flush=True)
        if not args.dry_run:
            subprocess.run(full_command, cwd=native_cwd, check=True)


if __name__ == '__main__':
    main()
