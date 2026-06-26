"""Diagnose a distilled STUDENT: how well does its legal-argmax match the TEACHER's?

Distinguishes two failure modes for a small distilled policy that plays badly:
  (a) HIGH argmax-match + bad gameplay  -> distribution shift (student errs once,
      lands off the teacher's state distribution where it has no signal, spirals).
      Fix = DAgger (relabel the student's OWN states) / more diverse corpus.
  (b) LOW argmax-match                  -> student is just far from the teacher
      (capacity or not enough epochs). Fix = more epochs / bigger student.

Samples states from a tensor, compares legal top-move (and top-3 containment).

    python -m alphatrain.scripts.diag_student_match \
        --teacher alphatrain/data/pillar3k_r3_dw3_T0.7_epoch_22.pt \
        --student alphatrain/data/pillar3k_small128_epoch_8.pt \
        --state-tensor alphatrain/data/distill_states.pt --n 20000
"""
import argparse
import numpy as np
import torch

from alphatrain.model import AlphaTrainNet
from alphatrain.dataset import TensorDatasetGPU
from alphatrain.mcts import _legal_priors_jit


def load(path, dev):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    st = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    if any(k.startswith('_orig_mod.') for k in st):
        st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    ch = st['stem.0.weight'].shape[0]
    nb = sum(1 for k in st if k.endswith('.conv1.weight') and k.startswith('blocks.'))
    m = AlphaTrainNet(num_blocks=nb, channels=ch).to(dev)
    m.load_state_dict(st, strict=False)
    m.train(False)
    return (m.half() if dev.type in ('mps', 'cuda') else m), nb, ch


def top_moves(net, dtype, ds, idx, K=5):
    """Return per-state list of legal top-K flat move indices (best first)."""
    res = []
    for s in range(0, len(idx), 4096):
        b = idx[s:s + 4096]
        obs = ds._build_obs_core(ds.boards[b], next_pos=ds.next_pos[b],
                                 next_col=ds.next_col[b], n_next=ds.n_next[b])
        with torch.no_grad():
            out = net(obs.to(dtype))
            lg = (out[0] if isinstance(out, tuple) else out).float().cpu().numpy()
        bd = ds.boards[b].cpu().numpy().astype(np.int8)
        for i in range(len(b)):
            k, fi, pr = _legal_priors_jit(bd[i], lg[i], K)
            res.append([int(x) for x in fi[:k]] if k > 0 else [])
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', required=True)
    p.add_argument('--student', required=True)
    p.add_argument('--state-tensor', required=True)
    p.add_argument('--n', type=int, default=20000)
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)

    teacher, tnb, tch = load(a.teacher, dev)
    student, snb, sch = load(a.student, dev)
    print(f"teacher {tnb}b x {tch}ch | student {snb}b x {sch}ch", flush=True)

    ds = TensorDatasetGPU(a.state_tensor, augment=False, color_augment=False,
                          augment_factor=1, device=a.device)
    N = ds.boards.shape[0]
    rng = np.random.default_rng(0)
    idx = torch.from_numpy(np.sort(rng.choice(N, size=min(a.n, N), replace=False))).to(dev)

    tdt = next(teacher.parameters()).dtype
    sdt = next(student.parameters()).dtype
    T = top_moves(teacher, tdt, ds, idx)
    S = top_moves(student, sdt, ds, idx)

    top1 = top1_in_t3 = t1_in_s3 = both_legal = 0
    n = 0
    for t, s in zip(T, S):
        if not t or not s:
            continue
        n += 1
        both_legal += 1
        if s[0] == t[0]:
            top1 += 1
        if s[0] in t[:3]:
            top1_in_t3 += 1
        if t[0] in s[:3]:
            t1_in_s3 += 1
    print(f"\nstates compared: {n:,}", flush=True)
    print(f"  student top-1 == teacher top-1 : {100*top1/n:5.1f}%  "
          f"(argmax agreement — what greedy play uses)", flush=True)
    print(f"  student top-1 in teacher top-3 : {100*top1_in_t3/n:5.1f}%", flush=True)
    print(f"  teacher top-1 in student top-3 : {100*t1_in_s3/n:5.1f}%", flush=True)
    print(f"\nREAD: >~85% top-1 -> student learned the policy; bad gameplay = distribution "
          f"shift (DAgger). <~65% -> far from teacher (more epochs / bigger student).",
          flush=True)


if __name__ == '__main__':
    main()
