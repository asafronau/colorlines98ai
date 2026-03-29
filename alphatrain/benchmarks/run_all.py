"""Run all AlphaTrain benchmarks.

Usage:
    python -m alphatrain.benchmarks.run_all
    python -m alphatrain.benchmarks.run_all --tensor-file data/alphatrain_v1.pt
"""

import argparse
import subprocess
import sys


BENCHMARKS = [
    ('observation', 'alphatrain.benchmarks.bench_observation', []),
    ('model', 'alphatrain.benchmarks.bench_model', []),
    ('collate', 'alphatrain.benchmarks.bench_collate', ['--tensor-file']),
    ('pairwise_collate', 'alphatrain.benchmarks.bench_pairwise_collate', ['--tensor-file']),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor-file', default='alphatrain/data/alphatrain_pairwise.pt')
    args = p.parse_args()

    print("=" * 60)
    print("AlphaTrain Benchmark Suite")
    print("=" * 60)

    for name, module, extra_args in BENCHMARKS:
        print(f"\n--- {name} ---")
        cmd = [sys.executable, '-m', module]
        if '--tensor-file' in extra_args:
            cmd += ['--tensor-file', args.tensor_file]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")

    print("\n" + "=" * 60)
    print("Done")


if __name__ == '__main__':
    main()
