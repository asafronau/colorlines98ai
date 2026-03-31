"""Centralized GPU inference server for parallel MCTS.

One GPU process handles all NN forward passes. CPU workers send batches of
observations via shared memory (zero-copy). Queues carry only integer signals.

Architecture:
    Workers write obs to shared memory → put (slot_id, count) on queue →
    GPU reads from shared memory, batches, evaluates →
    writes results to shared memory → signals worker via response queue.
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

    def __init__(self, model_path, num_workers, device=None, max_batch_per_worker=MAX_BATCH):
        self.model_path = model_path
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
                  self.request_queue, self.response_queues),
            daemon=True)
        self._process.start()

    def make_client(self, slot_id):
        """Create an InferenceClient for worker slot_id."""
        return InferenceClient(
            slot_id, self.obs_buf, self.pol_buf, self.val_buf,
            self.request_queue, self.response_queues[slot_id])

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
              request_queue, response_queues):
    """GPU inference process main loop."""
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

    # FP16 inference: ~2x faster forward pass on MPS
    net = net.half()

    # JIT trace for forward pass (~10-15% faster, no control flow in ResNet)
    # Keep original net for predict_value (not part of forward)
    net_traced = torch.jit.trace(net, torch.randn(1, 18, 9, 9, device=device).half())

    # Pre-allocate GPU-side batch buffer in fp16
    max_total = num_workers * max_batch
    gpu_obs = torch.empty(max_total, 18, 9, 9, device=device, dtype=torch.float16)

    total_evals = 0
    total_batches = 0
    t_start = time.time()

    print(f"GPU inference server ready ({device_str}, fp16+jit, "
          f"{num_workers} slots, max_batch={max_batch})", flush=True)

    # Cap GPU batch — too large hurts per-eval latency on MPS
    GPU_BATCH_CAP = 128

    while True:
        # Block on first request
        item = request_queue.get()
        if item is None:
            break
        pending = [item]  # list of (slot_id, count)
        pending_obs = item[1]

        # Drain additional pending requests (cap total obs)
        while pending_obs < GPU_BATCH_CAP:
            try:
                item = request_queue.get_nowait()
                if item is None:
                    request_queue.put(None)
                    break
                pending.append(item)
                pending_obs += item[1]
            except Empty:
                break

        # Gather obs from shared memory → fp16 GPU tensor
        total_count = 0
        for slot_id, count in pending:
            gpu_obs[total_count:total_count + count] = torch.from_numpy(
                obs_buf[slot_id, :count]).half()
            total_count += count

        # One forward pass (fp16, JIT traced)
        with torch.inference_mode():
            pol_logits, val_logits = net_traced(gpu_obs[:total_count])
            values = net.predict_value(val_logits, max_val=max_score)

        # Write results to shared memory (cast back to fp32 for workers)
        pol_cpu = pol_logits.float().cpu()
        val_cpu = values.float().cpu()
        offset = 0
        for slot_id, count in pending:
            pol_buf[slot_id, :count] = pol_cpu[offset:offset + count].numpy()
            val_buf[slot_id, :count] = val_cpu[offset:offset + count].numpy()
            offset += count
            response_queues[slot_id].put(1)

        total_evals += total_count
        total_batches += 1
        if total_batches % 200 == 0:
            elapsed = time.time() - t_start
            avg_bs = total_evals / total_batches
            print(f"  [GPU] {total_evals} evals, {total_batches} batches "
                  f"(avg bs={avg_bs:.1f}, {total_evals/elapsed:.0f} evals/s)",
                  flush=True)

    obs_shm.close()
    pol_shm.close()
    val_shm.close()
