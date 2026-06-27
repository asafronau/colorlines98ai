"""Export a PolicyNet to the C++ engine's binary format + a golden test vector.

Binary format (little-endian), used for both weights.bin and golden.bin:
    magic 'CLNW'
    uint32 num_tensors
    per tensor:
        uint32 name_len, <name bytes>,
        uint32 ndim, int32[ndim] dims,
        float32[prod(dims)] data   (row-major / C-contiguous)

The C++ side (LoadBlob in net.cc) reads exactly this. The golden lets the C++
forward pass be checked against PyTorch bit-close at every milestone.

    python -m alphatrain.inference_cpp.export_weights \
        --model alphatrain/data/pillar3k_small128_epoch_15.pt
"""
import argparse, os, struct
import numpy as np
import torch

from alphatrain.model import PolicyNet
from alphatrain.dataset import TensorDatasetGPU

MAGIC = b'CLNW'


def write_blob(path, tensors: dict):
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', len(tensors)))
        for name, arr in tensors.items():
            a = np.ascontiguousarray(arr, dtype='<f4')
            nb = name.encode('utf-8')
            f.write(struct.pack('<I', len(nb)))
            f.write(nb)
            f.write(struct.pack('<I', a.ndim))
            f.write(struct.pack('<%di' % a.ndim, *a.shape))
            f.write(a.tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3k_small128_epoch_15.pt')
    p.add_argument('--state-tensor', default='alphatrain/data/distill_states.pt')
    p.add_argument('--outdir', default='alphatrain/inference_cpp/data')
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    ck = torch.load(a.model, map_location='cpu', weights_only=False)
    st = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    if any(k.startswith('_orig_mod.') for k in st):
        st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    ch = st['stem.0.weight'].shape[0]
    nblocks = sum(1 for k in st if k.endswith('.conv1.weight') and k.startswith('blocks.'))
    m = PolicyNet(num_blocks=nblocks, channels=ch)
    m.load_state_dict(st)
    m.train(False)

    # weights blob (drop the integer BN counters; keep everything else as fp32)
    weights = {k: v.float().numpy() for k, v in st.items() if 'num_batches_tracked' not in k}
    write_blob(f'{a.outdir}/weights.bin', weights)

    # golden test vector from one real state
    ds = TensorDatasetGPU(a.state_tensor, augment=False, color_augment=False,
                          augment_factor=1, device='cpu')
    obs = ds._build_obs_core(ds.boards[0:1], next_pos=ds.next_pos[0:1],
                             next_col=ds.next_col[0:1], n_next=ds.n_next[0:1])
    cap = {}
    h = m.stem[0].register_forward_hook(
        lambda mod, i, o: cap.__setitem__('stem_conv_out', o.detach()))
    with torch.no_grad():
        logits = m(obs.float())
    h.remove()
    write_blob(f'{a.outdir}/golden.bin', {
        'obs': obs[0].numpy(),                       # (18, 9, 9)
        'stem_conv_out': cap['stem_conv_out'][0].numpy(),  # (C, 9, 9)
        'logits': logits[0].numpy(),                 # (6561,)
    })

    print(f'arch: {nblocks}b x {ch}ch')
    print(f'wrote {a.outdir}/weights.bin ({len(weights)} tensors)')
    print(f'wrote {a.outdir}/golden.bin: obs{tuple(obs[0].shape)} '
          f'stem_conv_out{tuple(cap["stem_conv_out"][0].shape)} logits{tuple(logits[0].shape)}')


if __name__ == '__main__':
    main()
