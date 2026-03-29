# AlphaTrain — Benchmark History

Track throughput across approaches to catch regressions.
Colab limit: ~24h. Training must complete within this window.

## Collate Throughput (samples/s)

| Date | Approach | MPS (M5 Max) | CUDA (A100) | Batch | Notes |
|------|----------|-------------|-------------|-------|-------|
| 2026-03-28 | Pillar 1 standard collate | 44K | 31K | 4096 | baseline, 18ch obs |
| 2026-03-29 | Pillar 2a TD standard collate | 44K | 16K* | 4096 | *different A100 alloc |
| 2026-03-29 | Pillar 2b pairwise collate v1 | 20K | TBD | 4096 | 2.2x overhead (3x obs, 2 CPU trips) |
| 2026-03-29 | Pillar 2b pairwise collate v2 | 27K | TBD | 4096 | 1.4x overhead (fused obs + minimal CPU buf) |
| 2026-03-29 | Pillar 2b pairwise collate v3 | 26K | TBD | 4096 | 1.7x MPS (GPU line potentials, no CPU sync) |
| 2026-03-29 | Pillar 2b pairwise collate v4 | **52K** | TBD | 4096 | **1.5x** (+ GPU component area via shifts) |
| 2026-03-29 | Standard collate v4 | **78K** | TBD | 4096 | GPU line pot + shift component area |

## Training Wall Clock

| Date | Approach | GPU | Epochs | Time | s/s | Notes |
|------|----------|-----|--------|------|-----|-------|
| 2026-03-28 | Pillar 1 (game_score) | A100 | 20 | 1.8h | 31K | no warmup |
| 2026-03-29 | Pillar 2a (TD γ=0.99) | A100 | 30 | 5.2h | 16K | warmup+cosine |
| 2026-03-29 | Pillar 2b (pairwise) | TBD | 30 | est ~10h | ~8K? | 2x forward passes per batch |

## Estimated Training Time Formula

```
time_hours = (n_samples × augment_factor × epochs) / (throughput_s_per_s × 3600)
```

For pairwise with 10.5M effective samples, 30 epochs, 8K s/s CUDA:
```
(10_500_000 × 30) / (8000 × 3600) = 10.9h
```

If throughput < 5K s/s on CUDA → training > 17.5h → dangerously close to 24h limit.

## Observation Benchmarks

| Date | Benchmark | Result | Target |
|------|-----------|--------|--------|
| 2026-03-28 | Single obs build | 0.8µs | <1ms |
| 2026-03-28 | Batch obs build (200) | 0.2ms | <1ms |
| 2026-03-28 | Model forward (batch=50, fp16, MPS) | 3.7ms | <10ms |
