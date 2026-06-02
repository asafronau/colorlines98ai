"""Is the per-step .cpu() cost payload-bound or fixed-overhead-bound?

Decides the optimization: if payload-bound, doing argmax/top-k ON the GPU and
transferring only tiny results gives a big win. If overhead-bound (per-call
sync), we must reduce the NUMBER of syncs (vectorize the game step on GPU) or
accept it. Times the same shapes the rollout uses (B=128, 6561 logits).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

dev = torch.device('mps' if torch.backends.mps.is_available()
                   else 'cuda' if torch.cuda.is_available() else 'cpu')
B, M = 128, 6561
ITERS = 300


def sync():
    if dev.type == 'mps':
        torch.mps.synchronize()
    elif dev.type == 'cuda':
        torch.cuda.synchronize()


def timeit(label, fn):
    x = torch.randn(B, M, device=dev, dtype=torch.float16)
    for _ in range(20):
        fn(x)
    sync()
    t = time.perf_counter()
    for _ in range(ITERS):
        fn(x)
    sync()
    dt = (time.perf_counter() - t) / ITERS * 1000
    print(f"  {label:<46} {dt:6.2f} ms/call", flush=True)


print(f"device={dev}  B={B} logits={M}  ({ITERS} iters)\n", flush=True)
print("CURRENT path:", flush=True)
timeit(".float().cpu().numpy()  [full 128x6561 fp32]",
       lambda x: x.float().cpu().numpy())
print("\nGPU-side selection (transfer only the result):", flush=True)
timeit(".argmax(1).cpu()        [128 int64]",
       lambda x: x.argmax(1).cpu().numpy())
timeit(".topk(64,1).cpu x2      [128x64 val+idx]",
       lambda x: (lambda r: (r.values.cpu().numpy(), r.indices.cpu().numpy()))(x.topk(64, 1)))
timeit("masked argmax on GPU + .cpu()  [reshape+mask+argmax]",
       lambda x: x.reshape(B, 81, 81).amax(2).argmax(1).cpu().numpy())
print("\nGPU compute only (no transfer):", flush=True)
timeit("argmax(1) [no .cpu]", lambda x: x.argmax(1))
timeit("just sync (forward already done)", lambda x: None)
