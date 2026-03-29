"""Benchmark model forward pass speed."""

import time
import torch
from alphatrain.model import AlphaTrainNet, count_parameters


def bench_forward(device_name, batch_size=50, fp16=False, channels_last=False,
                   num_blocks=10, channels=256):
    device = torch.device(device_name)
    model = AlphaTrainNet(num_blocks=num_blocks, channels=channels).to(device)
    if fp16:
        model = model.half()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    dtype = torch.float16 if fp16 else torch.float32
    x = torch.randn(batch_size, 18, 9, 9, dtype=dtype, device=device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        for _ in range(10):
            _ = model(x)
    if device_name == 'mps':
        torch.mps.synchronize()

    N = 50
    with torch.inference_mode():
        t0 = time.perf_counter()
        for _ in range(N):
            _ = model(x)
        if device_name == 'mps':
            torch.mps.synchronize()
        t1 = time.perf_counter()

    ms = (t1 - t0) / N * 1000
    label = f"{device_name} batch={batch_size}"
    if fp16:
        label += " fp16"
    if channels_last:
        label += " NHWC"
    print(f"  {label}: {ms:.1f}ms ({batch_size/ms*1000:.0f} samples/s)")
    return ms


if __name__ == '__main__':
    model = AlphaTrainNet()
    print(f"Model: {count_parameters(model):,} parameters")
    print()

    print("=== CPU ===")
    bench_forward('cpu', batch_size=1)
    bench_forward('cpu', batch_size=50)

    if torch.backends.mps.is_available():
        print("\n=== MPS ===")
        bench_forward('mps', batch_size=50)
        bench_forward('mps', batch_size=50, fp16=True)
        bench_forward('mps', batch_size=50, channels_last=True)
        bench_forward('mps', batch_size=50, fp16=True, channels_last=True)
        bench_forward('mps', batch_size=200)
        bench_forward('mps', batch_size=200, fp16=True)
