"""GPU inference server: many CPU workers, one GPU process.

Each worker writes observations to its slot in shared memory and signals
via a request queue. The GPU process gathers up to N×B requests, runs a
single forward pass, scatters policy logits back to per-worker shared
memory, and signals each worker's response queue.

PolicyNet only — leaf values come from the feature evaluator inside MCTS,
not from the model. val_buf is preserved in the protocol but always zero.
"""
import time
import numpy as np
import torch
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from queue import Empty

OBS_SHAPE = (18, 9, 9)
OBS_SIZE = 18 * 9 * 9
POL_SIZE = 6561
MAX_BATCH = 16     # default per-worker batch cap


class InferenceClient:
    """Worker-side handle. Writes obs to shared memory, signals via queue."""
    __slots__ = ('slot_id', 'obs_buf', 'pol_buf', 'val_buf',
                 'request_queue', 'response_queue')

    def __init__(self, slot_id, obs_buf, pol_buf, val_buf,
                 request_queue, response_queue):
        self.slot_id = slot_id
        self.obs_buf = obs_buf
        self.pol_buf = pol_buf
        self.val_buf = val_buf
        self.request_queue = request_queue
        self.response_queue = response_queue

    def evaluate(self, obs_np):
        """Send single observation, wait for (policy_logits, value).

        value is always 0.0 — leaf values come from the feature evaluator
        on the worker side. The val_buf field is kept in the protocol so
        the existing shared-memory layout is unchanged.
        """
        self.obs_buf[self.slot_id, 0] = obs_np
        self.request_queue.put((self.slot_id, 1))
        self.response_queue.get()
        return self.pol_buf[self.slot_id, 0].copy(), \
            float(self.val_buf[self.slot_id, 0])

    def evaluate_batch(self, obs_batch_np, count):
        """Send batch of observations, wait for results.

        Returns (pol_np, val_np) views into shared memory. val_np is zero;
        callers should ignore it and compute leaf values from features.
        """
        self.obs_buf[self.slot_id, :count] = obs_batch_np[:count]
        self.request_queue.put((self.slot_id, count))
        self.response_queue.get()
        return self.pol_buf[self.slot_id, :count], \
            self.val_buf[self.slot_id, :count]


class InferenceServer:
    """Manages shared memory and the GPU inference subprocess."""

    def __init__(self, model_path, num_workers, device=None,
                 max_batch_per_worker=MAX_BATCH, use_compile=False,
                 value_head_path=None, fp16=True):
        self.model_path = model_path
        self.use_compile = use_compile
        self.value_head_path = value_head_path
        self.fp16 = fp16
        self.num_workers = num_workers
        self.max_batch = max_batch_per_worker

        if device is None:
            if torch.backends.mps.is_available():
                self.device_str = 'mps'
            elif torch.cuda.is_available():
                self.device_str = 'cuda'
            else:
                self.device_str = 'cpu'
        else:
            self.device_str = str(device)

        N, B = num_workers, max_batch_per_worker
        self._obs_shm = SharedMemory(create=True, size=max(1, N * B * OBS_SIZE * 4))
        self._pol_shm = SharedMemory(create=True, size=max(1, N * B * POL_SIZE * 4))
        self._val_shm = SharedMemory(create=True, size=max(1, N * B * 4))

        self.obs_buf = np.ndarray((N, B) + OBS_SHAPE,
                                   dtype=np.float32, buffer=self._obs_shm.buf)
        self.pol_buf = np.ndarray((N, B, POL_SIZE),
                                   dtype=np.float32, buffer=self._pol_shm.buf)
        self.val_buf = np.ndarray((N, B),
                                   dtype=np.float32, buffer=self._val_shm.buf)

        self.request_queue = Queue()
        self.response_queues = [Queue() for _ in range(num_workers)]
        self._process = None

    def start(self):
        """Launch the GPU inference process."""
        self._process = Process(
            target=_gpu_loop,
            args=(self.model_path, self.device_str,
                  self.num_workers, self.max_batch,
                  self._obs_shm.name, self._pol_shm.name, self._val_shm.name,
                  self.request_queue, self.response_queues,
                  self.use_compile, self.value_head_path, self.fp16),
            daemon=True)
        self._process.start()

    def make_client(self, slot_id):
        return InferenceClient(slot_id, self.obs_buf, self.pol_buf, self.val_buf,
                                self.request_queue, self.response_queues[slot_id])

    def is_alive(self):
        return self._process is not None and self._process.is_alive()

    def shutdown(self):
        self.request_queue.put(None)
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
        for shm in (self._obs_shm, self._pol_shm, self._val_shm):
            shm.close()
            shm.unlink()


def _gpu_loop(model_path, device_str, num_workers, max_batch,
              obs_shm_name, pol_shm_name, val_shm_name,
              request_queue, response_queues, use_compile=False,
              value_head_path=None, fp16=True):
    """GPU inference process main loop.

    Gathers requests from multiple workers using busy-poll, then runs
    a single forward pass on the combined batch. Scatters results back
    via shared memory, signals each worker's response queue.

    When `value_head_path` is set, loads the trained ValueHead, runs it
    on the same backbone features as the policy, computes the scalar
    leaf value V = Σ w_h · σ(logit_h), and writes V into val_buf.
    Otherwise val_buf stays zero and clients are expected to use the
    feature-value evaluator (or accept zero leaf values for testing).
    """
    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)
    N, B = num_workers, max_batch
    obs_buf = np.ndarray((N, B) + OBS_SHAPE, dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, POL_SIZE), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)

    device = torch.device(device_str)
    compute_dtype = torch.float16 if fp16 else torch.float32
    from alphatrain.evaluate import load_model
    net, _ = load_model(model_path, device)
    net = net.to(compute_dtype)

    # Optional NN value head — runs over the SAME backbone features as
    # the policy. We don't trace the head separately; we'll trace the
    # combined forward function below.
    value_head = None
    horizon_weights = None
    value_head_target_type = 'survival'
    value_head_type = 'value_head'
    if value_head_path is not None:
        from alphatrain.value_head import (
            load_any, DEFAULT_HORIZON_WEIGHTS)
        value_head, ckpt_meta, value_head_type = load_any(
            value_head_path, device=device)
        value_head = value_head.to(compute_dtype)
        value_head_target_type = ckpt_meta.get('target_type', 'survival')
        if value_head_type == 'spatial':
            horizon_weights = None
            print(f"  SpatialValueHead loaded from {value_head_path} "
                  f"(scalar V, no horizon weighting)", flush=True)
        elif value_head_target_type == 'density':
            density_weights = (0.5, 0.3, 0.2)
            horizon_weights = torch.tensor(
                density_weights, dtype=compute_dtype, device=device)
            print(f"  ValueHead (DENSITY mode) loaded from {value_head_path}, "
                  f"horizons={ckpt_meta.get('horizons')}, "
                  f"weights={density_weights}", flush=True)
        else:
            horizon_weights = torch.tensor(
                DEFAULT_HORIZON_WEIGHTS, dtype=compute_dtype, device=device)
            print(f"  ValueHead (survival mode) loaded from {value_head_path}",
                  flush=True)

    # CUDA-only: convert weights + activations to channels_last (NHWC).
    # ResNets on Ampere/Ada are 1.3-2x faster in NHWC because cuDNN
    # picks Tensor Core kernels optimized for that layout. MPS doesn't
    # benefit (and may regress); CPU is unaffected.
    use_channels_last = (device_str == 'cuda')
    mem_fmt = (torch.channels_last
               if use_channels_last
               else torch.contiguous_format)
    if use_channels_last:
        net = net.to(memory_format=torch.channels_last)

    dummy = torch.empty(1, 18, 9, 9,
                        device=device, dtype=compute_dtype,
                        memory_format=mem_fmt).normal_()

    # Forward callable: either policy-only or fused (policy + scalar V).
    if value_head is not None:
        # 'spatial' => raw scalar, no horizon weighting.
        # 'density' => regression outputs, weighted sum.
        # 'survival' => sigmoid then weighted sum.
        mode = ('spatial' if value_head_type == 'spatial'
                else ('density' if value_head_target_type == 'density'
                      else 'survival'))

        if mode == 'spatial':
            class _PolicyValueWrapper(torch.nn.Module):
                def __init__(self, net, head):
                    super().__init__()
                    self.net = net
                    self.head = head

                def forward(self, x):
                    pol, feats = self.net.forward_with_features(x)
                    scalar_V = self.head(feats).squeeze(-1)
                    return pol, scalar_V

            forward_module = _PolicyValueWrapper(net, value_head)
        else:
            is_density = (mode == 'density')

            class _PolicyValueWrapper(torch.nn.Module):
                def __init__(self, net, head, hw, density):
                    super().__init__()
                    self.net = net
                    self.head = head
                    self.register_buffer('hw', hw)
                    self.density = density

                def forward(self, x):
                    pol, feats = self.net.forward_with_features(x)
                    vh_out = self.head(feats)
                    if self.density:
                        scalar_V = (vh_out * self.hw).sum(dim=-1)
                    else:
                        vh_probs = torch.sigmoid(vh_out)
                        scalar_V = (vh_probs * self.hw).sum(dim=-1)
                    return pol, scalar_V

            forward_module = _PolicyValueWrapper(net, value_head,
                                                  horizon_weights, is_density)
        forward_module = forward_module.to(device).to(compute_dtype).eval()
        returns_value = True
    else:
        forward_module = net
        returns_value = False

    if use_compile and device_str == 'cuda':
        # torch.compile(reduce-overhead) is faster than jit.trace on CUDA
        # for the small repeated batches we issue here. Pays a 1-2 min
        # warm-up upfront. CUDA-only — MPS support is incomplete.
        print("  Compiling model with torch.compile(mode='reduce-overhead')...",
              flush=True)
        t0 = time.time()
        net_traced = torch.compile(forward_module,
                                   mode='reduce-overhead', fullgraph=False)
        with torch.inference_mode():
            for _ in range(3):
                net_traced(dummy)
        print(f"  Compile + warmup: {time.time()-t0:.1f}s", flush=True)
    else:
        net_traced = torch.jit.trace(forward_module, dummy)

    max_total = num_workers * max_batch
    gpu_obs = torch.empty(max_total, 18, 9, 9,
                          device=device, dtype=compute_dtype,
                          memory_format=mem_fmt)
    obs_staging = np.empty((max_total,) + OBS_SHAPE, dtype=np.float32)
    # When no NN value head is configured, val_buf stays zero forever
    # and clients are expected to use the feature-value MCTS evaluator.
    # When a head is configured, val_buf is rewritten on every batch.
    val_buf[:] = 0.0

    total_evals = 0
    total_gpu_batches = 0
    t_start = time.time()
    print(f"GPU inference server ready ({device_str}, "
          f"{'fp16' if fp16 else 'fp32'}, "
          f"{num_workers} slots, max_batch={max_batch})", flush=True)

    GPU_BATCH_CAP = num_workers * max_batch
    # Busy-poll gather window. 0.5ms is conservative; tunable in future.
    GATHER_TIMEOUT = 0.0005 if num_workers > 1 else 0.0
    GATHER_MIN_EVALS = min(48, num_workers * max_batch)

    _queue_get = request_queue.get
    _queue_get_nowait = request_queue.get_nowait

    while True:
        item = _queue_get()
        if item is None:
            break
        pending = [item]
        pending_evals = item[1]

        # Gather more requests during the busy-poll window.
        if GATHER_TIMEOUT > 0 and pending_evals < GATHER_MIN_EVALS:
            deadline = time.monotonic() + GATHER_TIMEOUT
            while pending_evals < GPU_BATCH_CAP:
                try:
                    item = _queue_get_nowait()
                    if item is None:
                        request_queue.put(None)
                        break
                    pending.append(item)
                    pending_evals += item[1]
                except Empty:
                    if time.monotonic() >= deadline:
                        break

        # Final drain.
        while pending_evals < GPU_BATCH_CAP:
            try:
                item = _queue_get_nowait()
                if item is None:
                    request_queue.put(None)
                    break
                pending.append(item)
                pending_evals += item[1]
            except Empty:
                break

        # Stage obs into a contiguous CPU buffer, then transfer to GPU once.
        total_count = 0
        for slot_id, count in pending:
            np.copyto(obs_staging[total_count:total_count + count],
                      obs_buf[slot_id, :count])
            total_count += count
        gpu_obs[:total_count] = torch.from_numpy(
            obs_staging[:total_count]).to(compute_dtype)

        with torch.inference_mode():
            if returns_value:
                pol_logits, scalar_V = net_traced(gpu_obs[:total_count])
            else:
                pol_logits = net_traced(gpu_obs[:total_count])
                scalar_V = None

        # Scatter policy logits back. If a value head ran on the server,
        # also scatter scalar V into val_buf; otherwise val_buf stays zero.
        pol_cpu = pol_logits.float().cpu().numpy()
        val_cpu = scalar_V.float().cpu().numpy() if scalar_V is not None else None
        offset = 0
        for slot_id, count in pending:
            np.copyto(pol_buf[slot_id, :count],
                      pol_cpu[offset:offset + count])
            if val_cpu is not None:
                np.copyto(val_buf[slot_id, :count],
                          val_cpu[offset:offset + count])
            offset += count
            response_queues[slot_id].put(1)

        total_evals += total_count
        total_gpu_batches += 1

        if total_gpu_batches % 10000 == 0:
            elapsed = time.time() - t_start
            avg_bs = total_evals / max(total_gpu_batches, 1)
            print(f"  [GPU] {total_evals} evals, {total_gpu_batches} fwd "
                  f"(avg bs={avg_bs:.1f}, {total_evals/elapsed:.0f} evals/s)",
                  flush=True)

    obs_shm.close()
    pol_shm.close()
    val_shm.close()
