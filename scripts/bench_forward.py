"""Forward-pass throughput vs batch size — the rollout is GPU-forward-bound
(see bench_sync.py: the per-step cost is the forward, surfaced at .cpu()).

If ms/call grows sub-linearly with batch (per-op MPS dispatch overhead being
amortized), then raising the rollout batch is a cheap, big win. Reports
boards/s so we can pick the operating batch.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from alphatrain.evaluate import load_model

dev = torch.device('mps' if torch.backends.mps.is_available()
                   else 'cuda' if torch.cuda.is_available() else 'cpu')
net, _ = load_model('alphatrain/data/pillar3b_epoch_20.pt', dev, fp16=True)
dtype = next(net.parameters()).dtype


def sync():
    if dev.type == 'mps':
        torch.mps.synchronize()
    elif dev.type == 'cuda':
        torch.cuda.synchronize()


print(f"device={dev} dtype={dtype}\n", flush=True)
print(f"{'batch':>6} {'ms/call':>9} {'boards/s':>10} {'speedup/b128':>13}", flush=True)
print('-' * 42, flush=True)
base = None
for B in [32, 64, 128, 256, 512, 1024, 2048]:
    x = torch.randn(B, 18, 9, 9, device=dev, dtype=dtype)
    with torch.no_grad():
        for _ in range(10):
            net(x)
        sync()
        t = time.perf_counter()
        for _ in range(50):
            net(x)
        sync()
    ms = (time.perf_counter() - t) / 50 * 1000
    bps = B / ms * 1000
    if B == 128:
        base = bps
    sp = f"{bps/base:.2f}x" if base else "-"
    print(f"{B:>6} {ms:>9.2f} {bps:>10.0f} {sp:>13}", flush=True)
