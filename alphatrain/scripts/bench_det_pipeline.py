"""Test pipelined deterministic processing.

Compare:
1. Sequential: for each request: copy_in -> forward -> copy_out
2. Pipelined: overlap copy_in[i+1] with copy_out[i]

The key insight: on MPS, the forward pass is async. We can queue the next
copy_in before extracting the previous result, so the GPU doesn't idle.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import time
import numpy as np
import torch


def main():
    model_path = 'alphatrain/data/alphatrain_td_best.pt'
    if not os.path.exists(model_path):
        print("Model not found", flush=True)
        return

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        print("No MPS", flush=True)
        return

    from alphatrain.evaluate import load_model
    net, max_score = load_model(model_path, device)
    net = net.half()
    dummy = torch.randn(1, 18, 9, 9, device=device).half()
    net_traced = torch.jit.trace(net, dummy)

    n_req = 16  # simulate 16 pending requests
    bs = 8
    POL_SIZE = 6561

    # Create obs arrays (simulating shared memory)
    obs_arrays = [np.random.randn(bs, 18, 9, 9).astype(np.float32) for _ in range(n_req)]
    pol_arrays = [np.empty((bs, POL_SIZE), dtype=np.float32) for _ in range(n_req)]
    val_arrays = [np.empty(bs, dtype=np.float32) for _ in range(n_req)]

    # Two GPU buffers for double-buffering
    gpu_obs_a = torch.empty(bs, 18, 9, 9, device=device, dtype=torch.float16)
    gpu_obs_b = torch.empty(bs, 18, 9, 9, device=device, dtype=torch.float16)

    n_rep = 30

    # Warmup
    for _ in range(5):
        gpu_obs_a[:] = torch.from_numpy(obs_arrays[0]).half()
        with torch.inference_mode():
            pol, val = net_traced(gpu_obs_a)
            v = net.predict_value(val, max_val=max_score)
        pol_arrays[0][:] = pol.float().cpu().numpy()
        val_arrays[0][:] = v.float().cpu().numpy()

    # Method 1: Sequential (current approach)
    t_seq = 0
    for rep in range(n_rep):
        t0 = time.perf_counter()
        with torch.inference_mode():
            for i in range(n_req):
                gpu_obs_a[:] = torch.from_numpy(obs_arrays[i]).half()
                pol, val = net_traced(gpu_obs_a)
                v = net.predict_value(val, max_val=max_score)
                pol_arrays[i][:] = pol.float().cpu().numpy()
                val_arrays[i][:] = v.float().cpu().numpy()
        t_seq += time.perf_counter() - t0

    seq_per_req = t_seq / n_rep / n_req
    print(f"Sequential:  {seq_per_req*1e6:.0f} us/req, "
          f"{bs*n_req*n_rep/t_seq:.0f} evals/s", flush=True)

    # Method 2: Pipelined -- overlap copy_in[i+1] with forward[i]
    # The idea: submit copy_in as a non-blocking op, then extract previous result
    t_pipe = 0
    for rep in range(n_rep):
        t0 = time.perf_counter()
        with torch.inference_mode():
            # Prime: copy first obs
            gpu_a = gpu_obs_a
            gpu_b = gpu_obs_b
            gpu_a[:] = torch.from_numpy(obs_arrays[0]).half()

            for i in range(n_req):
                # Forward current
                pol, val = net_traced(gpu_a)
                v = net.predict_value(val, max_val=max_score)

                # While GPU might still be computing, queue next copy_in
                if i + 1 < n_req:
                    gpu_b[:] = torch.from_numpy(obs_arrays[i + 1]).half()

                # Now extract results (this syncs the GPU)
                pol_arrays[i][:] = pol.float().cpu().numpy()
                val_arrays[i][:] = v.float().cpu().numpy()

                # Swap buffers
                gpu_a, gpu_b = gpu_b, gpu_a

        t_pipe += time.perf_counter() - t0

    pipe_per_req = t_pipe / n_rep / n_req
    print(f"Pipelined:   {pipe_per_req*1e6:.0f} us/req, "
          f"{bs*n_req*n_rep/t_pipe:.0f} evals/s, "
          f"{t_seq/t_pipe:.2f}x faster", flush=True)

    # Method 3: Batch forward but split results
    # Process all 16 requests in one forward of bs=128 (NOT deterministic,
    # but measures theoretical maximum)
    all_obs = np.concatenate(obs_arrays, axis=0)  # (128, 18, 9, 9)
    gpu_all = torch.empty(n_req * bs, 18, 9, 9, device=device, dtype=torch.float16)

    t_batch = 0
    for rep in range(n_rep):
        t0 = time.perf_counter()
        gpu_all[:] = torch.from_numpy(all_obs).half()
        with torch.inference_mode():
            pol, val = net_traced(gpu_all)
            v = net.predict_value(val, max_val=max_score)
        pol_cpu = pol.float().cpu().numpy()
        val_cpu = v.float().cpu().numpy()
        for i in range(n_req):
            pol_arrays[i][:] = pol_cpu[i*bs:(i+1)*bs]
            val_arrays[i][:] = val_cpu[i*bs:(i+1)*bs]
        t_batch += time.perf_counter() - t0

    batch_per_req = t_batch / n_rep / n_req
    print(f"Batched(128):{batch_per_req*1e6:.0f} us/req, "
          f"{bs*n_req*n_rep/t_batch:.0f} evals/s, "
          f"{t_seq/t_batch:.2f}x faster", flush=True)


if __name__ == '__main__':
    main()
