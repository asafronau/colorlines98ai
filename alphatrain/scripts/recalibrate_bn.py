"""Recalibrate BatchNorm running stats on V12-only data.

Diagnostic for the concat-batch BN-contamination bug in train_path_b.py:
oracle observations were mixed into the V12 forward pass, so BN running_mean
and running_var drifted toward the crisis-heavy oracle distribution. This
script loads a checkpoint, runs N V12-only forward passes in train mode (no
backprop, no optimizer), and saves a new checkpoint with updated BN stats.

Compare eval gameplay on the original vs recalibrated checkpoint. If the
recalibrated one is materially stronger, BN contamination was the cause of
C's late regression.

Usage:
    python -m alphatrain.scripts.recalibrate_bn \\
        --checkpoint alphatrain/data/c_smoke_epoch_11.pt \\
        --tensor-file alphatrain/data/selfplay.pt \\
        --output alphatrain/data/c_smoke_epoch_11_bn.pt \\
        --n-batches 500 --batch-size 4096
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.model import AlphaTrainNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tensor-file', required=True,
                   help='V12 self-play tensor (build_expert_v2_tensor.py output).')
    p.add_argument('--output', required=True)
    p.add_argument('--n-batches', type=int, default=500,
                   help='Number of V12 batches to forward through the model. '
                        '500 with batch=4096 covers ~2M states; BN momentum=0.1 '
                        'replaces running stats within 100 batches.')
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--device', default=None)
    p.add_argument('--bn-momentum', type=float, default=None,
                   help='Override BN momentum during recalibration. Default '
                        'keeps the value baked into the checkpoint.')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    # ── Load checkpoint ──
    print(f"\nLoading {args.checkpoint}...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model = AlphaTrainNet(num_blocks=args.num_blocks,
                          channels=args.channels).to(device)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                 if k in model_state and v.shape == model_state[k].shape}
    skipped = [k for k in state if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    epoch = ckpt.get('epoch', '?')
    val_loss = ckpt.get('val_loss', float('nan'))
    print(f"  epoch={epoch}  val_loss={val_loss:.4f}", flush=True)
    if skipped:
        print(f"  skipped {len(skipped)} keys", flush=True)

    # Snapshot BN stats BEFORE recalibration for the diff print
    bn_before = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            bn_before[name] = (mod.running_mean.detach().clone(),
                                mod.running_var.detach().clone())

    if args.bn_momentum is not None:
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.momentum = args.bn_momentum
        print(f"  BN momentum set to {args.bn_momentum}", flush=True)

    # ── V12 data loader (unaugmented, train-mode pass) ──
    # Use only train split; we just want representative V12 states.
    print(f"\nLoading V12 tensor {args.tensor_file}...", flush=True)
    train_set, _ = TensorDatasetGPU.make_train_val_split(
        args.tensor_file,
        val_split=0.05,
        augment=False,          # don't augment — we want raw V12 stats
        color_augment=False,
        augment_factor=1,
        device=str(device),
        seed=args.seed,
    )
    print(f"  train states available: {len(train_set):,}", flush=True)

    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=train_set.collate)

    # ── Recalibration: forward in train mode, NO backprop ──
    model.train(True)
    print(f"\nRunning {args.n_batches} V12-only forward passes "
          f"(train mode, no backprop)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        seen = 0
        for bi, batch in enumerate(loader):
            obs, _ = batch
            obs = obs.to(device, non_blocking=True)
            _ = model(obs)
            seen += obs.shape[0]
            if (bi + 1) % 50 == 0:
                print(f"  [{bi+1}/{args.n_batches}] {seen:,} states "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if bi + 1 >= args.n_batches:
                break
    print(f"  done in {time.time()-t0:.0f}s, "
          f"saw {seen:,} states", flush=True)

    # ── BN diff print ──
    print(f"\nBN running stats diff (sampling 3 layers):")
    sample_layers = list(bn_before.keys())[:3]
    for name in sample_layers:
        mod = dict(model.named_modules())[name]
        m_old, v_old = bn_before[name]
        m_new = mod.running_mean.detach()
        v_new = mod.running_var.detach()
        dm = (m_new - m_old).abs().mean().item()
        dv = (v_new - v_old).abs().mean().item()
        print(f"  {name}: |Δmean|={dm:.4f}, |Δvar|={dv:.4f}",
              flush=True)

    # ── Save ──
    model.train(False)
    out_ckpt = copy.deepcopy(ckpt)
    out_ckpt['model'] = model.state_dict()
    out_ckpt['bn_recalibrated'] = {
        'source_checkpoint': args.checkpoint,
        'tensor_file': args.tensor_file,
        'n_batches': args.n_batches,
        'batch_size': args.batch_size,
        'states_seen': seen,
    }
    torch.save(out_ckpt, args.output)
    print(f"\nSaved {args.output} "
          f"({os.path.getsize(args.output)/1e6:.0f} MB)", flush=True)
    print(f"\nNext: run policy-only eval on {args.output} and compare to "
          f"{args.checkpoint}.", flush=True)


if __name__ == '__main__':
    main()
