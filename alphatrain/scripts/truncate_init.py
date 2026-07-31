"""Truncation warm-start: initialize a NARROWER student from a WIDER teacher.

Gemini's idea: copy the teacher's first N channels into the student instead of
random init, for a big head start. Implementation:
  1. Slice every parameter to the student's shape (first-N along each dim).
  2. Scale conv weights whose INPUT channels shrank (e.g. 256->128) by
     sqrt(in_teacher/in_student) to preserve activation variance.
  3. RECALIBRATE every BatchNorm's running_mean/var (the truncation shifts the
     activation statistics, so the teacher's stored stats are wrong) by running
     forward passes over data in train mode.

VERIFY the init's argmax-match to the teacher is well above random (~10%) before
spending GPU on training (diag_student_match.py).

    python -m alphatrain.scripts.truncate_init \
        --teacher alphatrain/data/pillar3k_r3_dw3_T0.7_epoch_22.pt \
        --out alphatrain/data/pillar3k_small128_truncinit.pt \
        --channels 128 --device mps
"""
import argparse
import numpy as np
import torch

from alphatrain.model import PolicyNet
from alphatrain.dataset import TensorDatasetGPU


def load_state(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    st = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    if any(k.startswith('_orig_mod.') for k in st):
        st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    return st


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--recal-tensor', default='alphatrain/data/distill_pillar3k.pt')
    p.add_argument('--recal-batches', type=int, default=200)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)

    tst = load_state(a.teacher)
    student = PolicyNet(num_blocks=a.num_blocks, channels=a.channels)
    sst = student.state_dict()

    scaled = 0
    new = {}
    for k, sp in sst.items():
        tp = tst[k]
        sl = tuple(slice(0, sp.shape[d]) for d in range(sp.ndim))
        w = tp[sl].clone().float()
        # conv weight (out,in,kh,kw): if INPUT channels shrank, scale by sqrt(ratio)
        if w.ndim == 4 and tp.shape[1] > sp.shape[1]:
            w *= (tp.shape[1] / sp.shape[1]) ** 0.5
            scaled += 1
        new[k] = w
    student.load_state_dict(new, strict=True)
    student = student.to(dev)
    print(f"Copied teacher->student first-{a.channels} channels "
          f"({scaled} convs √-scaled for reduced input).", flush=True)

    # ---- recalibrate BatchNorm running stats on the truncated activations ----
    for m in student.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
    ds = TensorDatasetGPU(a.recal_tensor, augment=False, color_augment=False,
                          augment_factor=1, device=a.device)
    N = ds.boards.shape[0]
    rng = np.random.default_rng(0)
    student.train(True)   # BN updates running stats from batch stats
    with torch.no_grad():
        for b in range(a.recal_batches):
            idx = torch.from_numpy(rng.choice(N, a.batch, replace=False)).to(dev)
            obs = ds._build_obs_core(ds.boards[idx], next_pos=ds.next_pos[idx],
                                     next_col=ds.next_col[idx], n_next=ds.n_next[idx])
            student(obs.float())
    student.train(False)
    print(f"Recalibrated BN over {a.recal_batches} batches of {a.batch}.", flush=True)

    torch.save({'model': student.state_dict(), 'max_score': 30000.0,
                'note': f'truncation warm-start from {a.teacher}'}, a.out)
    print(f"Saved {a.out}", flush=True)


if __name__ == '__main__':
    main()
