"""Centralized GPU inference server for parallel MCTS.

One GPU process handles all NN forward passes. CPU workers send batches of
observations via shared memory (zero-copy). Queues carry only integer signals.

Architecture:
    Workers write obs to shared memory -> put (slot_id, count) on queue ->
    GPU reads from shared memory, batches, evaluates ->
    writes results to shared memory -> signals worker via response queue.

Performance optimizations (2026-04-03):
    - Batched mode: adaptive gather window with busy-poll (up to 0.5ms) to
      accumulate requests from multiple workers into larger GPU batches
      (avg bs=90-128 vs previous avg bs=8, measured 20-32% throughput gain)
    - Contiguous obs staging buffer for single GPU transfer per batch
    - Batch-level CPU result transfer (.float().cpu()) then numpy scatter
"""

import time
import numpy as np
import torch
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from queue import Empty

OBS_SHAPE = (18, 9, 9)
OBS_SIZE = 18 * 9 * 9   # 1458 floats
POL_SIZE = 6561
MAX_BATCH = 16           # max leaves per worker batch


class InferenceClient:
    """Worker-side handle. Writes obs to shared memory, signals via queue."""
    __slots__ = ('slot_id', 'obs_buf', 'pol_buf', 'val_buf',
                 'request_queue', 'response_queue')

    def __init__(self, slot_id, obs_buf, pol_buf, val_buf,
                 request_queue, response_queue):
        self.slot_id = slot_id
        self.obs_buf = obs_buf      # view into shared memory
        self.pol_buf = pol_buf
        self.val_buf = val_buf
        self.request_queue = request_queue
        self.response_queue = response_queue

    def evaluate(self, obs_np):
        """Send single observation, wait for (policy_logits, value).

        Args:
            obs_np: numpy array (18, 9, 9) float32

        Returns:
            (policy_logits: np.ndarray (6561,), value: float)
        """
        self.obs_buf[self.slot_id, 0] = obs_np
        self.request_queue.put((self.slot_id, 1))
        self.response_queue.get()
        return self.pol_buf[self.slot_id, 0].copy(), float(self.val_buf[self.slot_id, 0])

    def evaluate_batch(self, obs_batch_np, count):
        """Send batch of observations, wait for results.

        Args:
            obs_batch_np: numpy array to write into shared obs_buf
            count: number of observations in the batch

        Returns:
            (pol_np, val_np) views into shared memory (valid until next call)
        """
        self.obs_buf[self.slot_id, :count] = obs_batch_np[:count]
        self.request_queue.put((self.slot_id, count))
        self.response_queue.get()
        return self.pol_buf[self.slot_id, :count], self.val_buf[self.slot_id, :count]


class InferenceServer:
    """Manages shared memory and GPU inference process."""

    def __init__(self, model_path, num_workers, device=None,
                 max_batch_per_worker=MAX_BATCH, value_model_path=None,
                 deterministic=False, value_mode=None,
                 ranking_head_path=None, use_compile=False):
        self.model_path = model_path
        self.value_model_path = value_model_path
        self.deterministic = deterministic
        self.value_mode = value_mode
        self.ranking_head_path = ranking_head_path
        self.use_compile = use_compile
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

        N = num_workers
        B = max_batch_per_worker

        # Shared memory: obs[N, B, 18, 9, 9], pol[N, B, 6561], val[N, B]
        self._obs_shm = SharedMemory(
            create=True, size=max(1, N * B * OBS_SIZE * 4))
        self._pol_shm = SharedMemory(
            create=True, size=max(1, N * B * POL_SIZE * 4))
        self._val_shm = SharedMemory(
            create=True, size=max(1, N * B * 4))

        self.obs_buf = np.ndarray(
            (N, B) + OBS_SHAPE, dtype=np.float32, buffer=self._obs_shm.buf)
        self.pol_buf = np.ndarray(
            (N, B, POL_SIZE), dtype=np.float32, buffer=self._pol_shm.buf)
        self.val_buf = np.ndarray(
            (N, B), dtype=np.float32, buffer=self._val_shm.buf)

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
                  self.value_model_path, self.deterministic,
                  self.value_mode, self.ranking_head_path,
                  self.use_compile),
            daemon=True)
        self._process.start()

    def make_client(self, slot_id):
        """Create an InferenceClient for worker slot_id."""
        return InferenceClient(
            slot_id, self.obs_buf, self.pol_buf, self.val_buf,
            self.request_queue, self.response_queues[slot_id])

    def is_alive(self):
        """Check if the GPU inference process is still running."""
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
              request_queue, response_queues, value_model_path=None,
              deterministic=False, value_mode=None,
              ranking_head_path=None, use_compile=False):
    """GPU inference process main loop.

    Two modes:
    - deterministic: process each worker's batch individually (bit-for-bit
      reproducible with local mode on MPS).
    - batched: gather requests from multiple workers into one large GPU batch
      for higher throughput (results are statistically equivalent but not
      bit-for-bit identical due to MPS batch-size non-determinism).
    """
    # Attach to shared memory
    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)

    N = num_workers
    B = max_batch
    obs_buf = np.ndarray((N, B) + OBS_SHAPE, dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, POL_SIZE), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)

    device = torch.device(device_str)
    from alphatrain.evaluate import load_model
    net, max_score = load_model(model_path, device)
    net = net.half()
    is_policy_only = getattr(net, 'policy_only', False)

    # Fail-fast: if the model has no value head AND the server has no
    # alternative value source, val_buf will be zeros — clients that try
    # to use it will silently train on garbage. Force the user to either
    # use a non-policy-only model, or configure an external value source
    # (value_model_path, value_mode, ranking_head_path).
    server_has_value_source = (value_model_path is not None
                               or value_mode is not None
                               or ranking_head_path is not None)
    if is_policy_only and not server_has_value_source:
        print("  WARNING: policy_only model loaded with no server-side value "
              "source configured. val_buf will be zeros — clients MUST use "
              "feature_weights_path or pass value_net to MCTS, otherwise "
              "search will run blind.", flush=True)

    dummy = torch.randn(1, 18, 9, 9, device=device).half()
    if use_compile and device_str == 'cuda':
        # torch.compile with reduce-overhead is the right mode for the
        # repeated small batches the inference server feeds (bs=8 per worker,
        # batched up to gpu_batch_cap across workers). First call is slow
        # (1-2 min warm-up), subsequent calls are faster than jit.trace.
        # CUDA-only: MPS support for torch.compile is incomplete.
        print("  Compiling model with torch.compile(mode='reduce-overhead')...",
              flush=True)
        import time as _t
        _t0 = _t.time()
        net_traced = torch.compile(net, mode='reduce-overhead', fullgraph=False)
        # Warm up so the first real inference doesn't pay compile cost
        with torch.inference_mode():
            for _ in range(3):
                net_traced(dummy)
        print(f"  Compile + warmup: {_t.time()-_t0:.1f}s", flush=True)
    else:
        net_traced = torch.jit.trace(net, dummy)

    # Separate ValueNet (if provided)
    value_net_traced = None
    if value_model_path:
        from alphatrain.model import ValueNet
        ckpt = torch.load(value_model_path, map_location='cpu', weights_only=False)
        vnet = ValueNet(in_channels=18,
                        num_blocks=ckpt['num_blocks'],
                        channels=ckpt['channels'],
                        num_value_bins=1)
        vnet.load_state_dict(ckpt['model'])
        vnet = vnet.to(device).half()
        value_net_traced = torch.jit.trace(vnet, dummy)
        print(f"  ValueNet: {ckpt['num_blocks']}b x {ckpt['channels']}ch, "
              f"acc={ckpt.get('accuracy', 0):.1f}%", flush=True)

    # Ranking head: loads backbone (non-JIT) + small ranking readout
    ranking_fn = None
    if ranking_head_path:
        from alphatrain.scripts.train_ranking_head import (
            RankingHead, SpatialRankingHead)
        rckpt = torch.load(ranking_head_path, map_location='cpu',
                           weights_only=False)
        if rckpt.get('spatial', False):
            rhead = SpatialRankingHead(
                channels=256,
                value_channels=rckpt.get('value_channels', 32),
                value_hidden=rckpt.get('value_hidden', 256))
        else:
            rhead = RankingHead(in_features=rckpt.get('in_features', 256),
                                hidden=rckpt.get('hidden', 0))
        rhead.load_state_dict(rckpt['head'])
        rhead = rhead.to(device).half()
        is_spatial = rckpt.get('spatial', False)
        # Load a non-JIT copy of backbone for feature extraction
        backbone_net, _ = load_model(model_path, device)
        backbone_net = backbone_net.half()
        backbone_net.eval()
        import torch.nn.functional as _F

        def _ranking_eval(obs_half, count):
            obs = obs_half[:count].clone()
            obs[:, 8:12] = 0  # zero fake next_balls to match training
            with torch.inference_mode():
                out = backbone_net.stem(obs)
                out = backbone_net.blocks(out)
                out = _F.relu(backbone_net.backbone_bn(out))
                if not is_spatial:
                    out = out.mean(dim=(2, 3))
                return torch.sigmoid(rhead(out).squeeze(-1))

        ranking_fn = _ranking_eval
        print(f"  RankingHead: hidden={rckpt.get('hidden', 0)}, "
              f"val_acc={100*rckpt.get('val_acc', 0):.1f}%", flush=True)

    # Value mode: 'zero', 'hash', 'iid:<std>', or None (use model head)
    if value_mode:
        print(f"  Value mode: {value_mode}", flush=True)

    # Pre-allocate GPU-side batch buffer in fp16
    max_total = num_workers * max_batch
    gpu_obs = torch.empty(max_total, 18, 9, 9, device=device, dtype=torch.float16)

    # Pre-allocate flat obs staging buffer for contiguous GPU transfer.
    # When gathering obs from multiple workers, we first copy to this
    # contiguous buffer, then do a single torch.from_numpy -> GPU transfer.
    obs_staging = np.empty((max_total,) + OBS_SHAPE, dtype=np.float32)

    total_evals = 0
    total_batches = 0
    total_gpu_batches = 0
    t_start = time.time()

    mode = "value_net" if value_model_path else \
           f"value_mode={value_mode}" if value_mode else "single-model"
    print(f"GPU inference server ready ({device_str}, fp16+jit, "
          f"{num_workers} slots, max_batch={max_batch}, {mode})", flush=True)

    def _compute_values(obs_half, count, pol_logits, val_logits):
        """Compute values based on value_mode setting."""
        if ranking_fn is not None:
            return ranking_fn(obs_half, count)
        if value_net_traced is not None:
            vnet_logits = value_net_traced(obs_half[:count])
            return torch.sigmoid(vnet_logits.squeeze(-1))
        if value_mode == 'zero':
            return torch.zeros(count, device=device)
        if value_mode and value_mode.startswith('iid:'):
            std = float(value_mode.split(':')[1])
            return torch.randn(count, device=device) * std + 100
        if value_mode == 'hash':
            # Deterministic per-state: hash the observation bytes
            obs_np = obs_half[:count].float().cpu().numpy()
            hashes = np.array([
                hash(obs_np[i].tobytes()) % (2**31) / (2**31)
                for i in range(count)], dtype=np.float32)
            return torch.from_numpy(hashes).to(device) * 200
        if val_logits is None:
            # policy_only model with no external value source — clients
            # using feature-value MCTS will ignore val_buf entirely.
            return torch.zeros(count, device=device)
        return net.predict_value(val_logits, max_val=max_score)

    GPU_BATCH_CAP = num_workers * max_batch  # no artificial cap

    # Adaptive gather: in batched mode with multiple workers, busy-poll
    # briefly after first request to accumulate a larger GPU batch.
    # This trades ~0.2-0.5ms latency for much better GPU utilization
    # (bs=32-128 vs bs=8 when processing each worker's request immediately).
    #
    # We use busy-polling (get_nowait + time.monotonic) instead of
    # get(timeout=) because the OS scheduler resolution on macOS is ~1ms,
    # which makes short timeouts unreliable.
    GATHER_TIMEOUT = 0.0005 if (not deterministic and num_workers > 1) else 0.0
    # Minimum batch to fill before processing.
    GATHER_MIN_EVALS = min(48, num_workers * max_batch)

    # Localize queue methods for hot loop
    _queue_get = request_queue.get
    _queue_get_nowait = request_queue.get_nowait

    while True:
        # Block on first request
        item = _queue_get()
        if item is None:
            break
        pending = [item]  # list of (slot_id, count)
        pending_evals = item[1]

        if deterministic:
            # Deterministic mode: drain whatever is available, no waiting.
            while True:
                try:
                    item = _queue_get_nowait()
                    if item is None:
                        request_queue.put(None)
                        break
                    pending.append(item)
                except Empty:
                    break

            # Per-request processing: each worker's batch individually.
            with torch.inference_mode():
                for slot_id, count in pending:
                    gpu_obs[:count] = torch.from_numpy(
                        obs_buf[slot_id, :count]).half()

                    out = net_traced(gpu_obs[:count])
                    if is_policy_only:
                        pol_logits, val_logits = out, None
                    else:
                        pol_logits, val_logits = out
                    values = _compute_values(gpu_obs, count, pol_logits, val_logits)

                    pol_buf[slot_id, :count] = pol_logits.float().cpu().numpy()
                    val_buf[slot_id, :count] = values.float().cpu().numpy()
                    response_queues[slot_id].put(1)
                    total_evals += count
                    total_gpu_batches += 1
        else:
            # Batched mode: gather requests from multiple workers, then
            # process in one large GPU batch for higher throughput.

            # Adaptive gather: busy-poll for more requests until we reach
            # the minimum batch threshold or the gather window expires.
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

            # Final drain: pick up any requests that arrived during the
            # gather window or are immediately available.
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

            # Stage obs into contiguous buffer, then single transfer to GPU
            total_count = 0
            for slot_id, count in pending:
                np.copyto(obs_staging[total_count:total_count + count],
                          obs_buf[slot_id, :count])
                total_count += count

            gpu_obs[:total_count] = torch.from_numpy(
                obs_staging[:total_count]).half()

            with torch.inference_mode():
                out = net_traced(gpu_obs[:total_count])
                if is_policy_only:
                    pol_logits, val_logits = out, None
                else:
                    pol_logits, val_logits = out
                values = _compute_values(
                    gpu_obs, total_count, pol_logits, val_logits)

            # Transfer GPU results to CPU in one shot, then scatter to SHM.
            # .float().cpu() is faster than copy_() on MPS.
            pol_cpu = pol_logits.float().cpu()
            val_cpu = values.float().cpu()

            # Scatter results to per-worker shared memory
            offset = 0
            pol_np = pol_cpu.numpy()
            val_np = val_cpu.numpy()
            for slot_id, count in pending:
                np.copyto(pol_buf[slot_id, :count],
                          pol_np[offset:offset + count])
                np.copyto(val_buf[slot_id, :count],
                          val_np[offset:offset + count])
                offset += count
                response_queues[slot_id].put(1)
            total_evals += total_count
            total_gpu_batches += 1

        total_batches += len(pending)
        if total_batches % 10000 == 0:
            elapsed = time.time() - t_start
            avg_bs = total_evals / max(total_gpu_batches, 1)
            print(f"  [GPU] {total_evals} evals, {total_gpu_batches} fwd "
                  f"(avg bs={avg_bs:.1f}, {total_evals/elapsed:.0f} evals/s)",
                  flush=True)

    obs_shm.close()
    pol_shm.close()
    val_shm.close()
