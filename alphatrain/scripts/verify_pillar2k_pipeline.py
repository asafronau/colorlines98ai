"""Verify the full Pillar 2k pipeline before Colab training.

Checks:
1. Tensor has all required fields (boards, val_targets, turns_remaining, pairs, etc.)
2. Dataset loads with all 2k flags (endgame, adversarial, etc.)
3. New model architecture builds correctly
4. Warm start from 2j handles shape mismatches (value head reinit)
5. One training step runs without error (forward + backward + optimizer)
6. Checkpoint save/load round-trips correctly
7. Load_model auto-detects new architecture
"""

import os
import sys
import time
import torch
import torch.nn.functional as F

def check(condition, msg):
    if condition:
        print(f"  OK: {msg}", flush=True)
    else:
        print(f"  FAIL: {msg}", flush=True)
        sys.exit(1)

def main():
    device = 'cpu'  # local verification, no GPU needed
    tensor_path = 'alphatrain/data/expert_v2_pairwise_g095.pt'
    model_path = 'alphatrain/data/pillar2j_best.pt'

    # ================================================================
    print("=" * 60, flush=True)
    print("1. TENSOR VERIFICATION", flush=True)
    print("=" * 60, flush=True)

    check(os.path.exists(tensor_path), f"Tensor exists: {tensor_path}")
    data = torch.load(tensor_path, weights_only=True, map_location='cpu')

    required_fields = [
        'boards', 'next_pos', 'next_col', 'n_next',
        'pol_indices', 'pol_values', 'pol_nnz', 'val_targets',
        'turns_remaining',  # NEW in 2j
        'good_boards', 'bad_boards', 'margins', 'pair_base_idx',
        'num_value_bins', 'max_score', 'gamma', 'n_pairs',
    ]
    for f in required_fields:
        check(f in data, f"Field '{f}' present")

    N = data['boards'].shape[0]
    check(data['boards'].shape == (N, 9, 9), f"boards shape: {data['boards'].shape}")
    check(data['val_targets'].shape == (N, 64), f"val_targets shape: {data['val_targets'].shape}")
    check(data['turns_remaining'].shape == (N,), f"turns_remaining shape: {data['turns_remaining'].shape}")
    check(data['good_boards'].shape[0] == N, f"good_boards count matches: {data['good_boards'].shape[0]}")
    check(float(data['max_score']) == 100.0, f"max_score=100: {data['max_score']}")
    check(float(data['gamma']) == 0.95, f"gamma=0.95: {data['gamma']}")
    check(int(data['num_value_bins']) == 64, f"num_value_bins=64: {data['num_value_bins']}")

    # Verify value targets sum to ~1 (valid probability distributions)
    vt_sums = data['val_targets'][:1000].sum(dim=-1)
    check((vt_sums - 1.0).abs().max() < 0.01, f"val_targets sum to 1.0 (max dev: {(vt_sums - 1.0).abs().max():.4f})")

    # Verify turns_remaining has endgame positions
    tr = data['turns_remaining']
    n_endgame = (tr <= 100).sum().item()
    check(n_endgame > 0, f"Endgame positions (<=100 turns): {n_endgame:,} ({100*n_endgame/N:.1f}%)")

    # Verify margins are positive
    check((data['margins'] >= 0).all(), f"All margins >= 0")

    print(f"\n  Summary: {N:,} states, {int(data['n_pairs']):,} pairs, "
          f"{n_endgame:,} endgame positions", flush=True)
    del data

    # ================================================================
    print("\n" + "=" * 60, flush=True)
    print("2. DATASET LOADING", flush=True)
    print("=" * 60, flush=True)

    from alphatrain.dataset import TensorDatasetGPU
    ds = TensorDatasetGPU(tensor_path, augment=True, device=device,
                          endgame_fraction=0.3, endgame_threshold=100,
                          adversarial_ranking=True)
    check(ds.max_score == 100.0, f"Dataset max_score: {ds.max_score}")
    check(ds.num_value_bins == 64, f"Dataset num_value_bins: {ds.num_value_bins}")
    check(ds.has_pairs, "Dataset has pairwise data")
    check(ds._endgame_indices is not None, f"Endgame indices loaded: {len(ds._endgame_indices):,}")
    check(ds.adversarial_ranking, "Adversarial ranking enabled")

    # Test collate_pairwise
    batch = list(range(32))
    obs, pol, val, good_obs, bad_obs, margin = ds.collate_pairwise(batch)
    check(obs.shape == (32, 18, 9, 9), f"obs shape: {obs.shape}")
    check(pol.shape == (32, 6561), f"pol shape: {pol.shape}")
    check(val.shape == (32, 64), f"val shape: {val.shape}")
    check(good_obs.shape == (32, 18, 9, 9), f"good_obs shape: {good_obs.shape}")
    check(bad_obs.shape == (32, 18, 9, 9), f"bad_obs shape: {bad_obs.shape}")
    check(margin.shape == (32,), f"margin shape: {margin.shape}")
    check((margin == 50.0).all(), f"Adversarial margins all 50.0: {margin[0].item()}")

    # Verify good and bad are actually different
    diff = (good_obs - bad_obs).abs().sum().item()
    check(diff > 0, f"Good and bad obs are different (diff={diff:.1f})")

    # ================================================================
    print("\n" + "=" * 60, flush=True)
    print("3. MODEL ARCHITECTURE (Pillar 2k)", flush=True)
    print("=" * 60, flush=True)

    from alphatrain.model import AlphaTrainNet
    model = AlphaTrainNet(num_blocks=10, channels=256,
                          num_value_bins=64,
                          value_channels=32, value_hidden=512, value_dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    n_value = sum(p.numel() for n, p in model.named_parameters() if 'value' in n)
    check(n_params > 13_000_000, f"Total params: {n_params:,}")
    check(n_value > 1_000_000, f"Value head params: {n_value:,}")

    # Test forward pass
    x = torch.randn(2, 18, 9, 9)
    pol_logits, val_logits = model(x)
    check(pol_logits.shape == (2, 6561), f"Policy output: {pol_logits.shape}")
    check(val_logits.shape == (2, 64), f"Value output: {val_logits.shape}")

    # Test predict_value
    pred = model.predict_value(val_logits, max_val=100.0)
    check(pred.shape == (2,), f"predict_value output: {pred.shape}")
    check((pred >= 0).all() and (pred <= 100).all(), f"Values in [0, 100]: {pred.tolist()}")

    # ================================================================
    print("\n" + "=" * 60, flush=True)
    print("4. WARM START FROM PILLAR 2j", flush=True)
    print("=" * 60, flush=True)

    if not os.path.exists(model_path):
        print(f"  SKIP: {model_path} not found (download from Drive first)", flush=True)
    else:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        state = ckpt['model']
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}

        model_state = model.state_dict()
        transferred = []
        skipped = []
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                transferred.append(k)
            else:
                skipped.append(k)

        check(len(transferred) > 0, f"Transferred {len(transferred)} layers")
        print(f"  Skipped (shape mismatch): {skipped}", flush=True)

        # Verify backbone and policy transfer, value head reinits
        backbone_transferred = [k for k in transferred
                                if k.startswith(('stem.', 'blocks.', 'backbone_', 'policy_'))]
        value_skipped = [k for k in skipped if 'value' in k]
        check(len(backbone_transferred) > 40, f"Backbone+policy layers transferred: {len(backbone_transferred)}")
        check(len(value_skipped) >= 4, f"Value head layers skipped (reinit): {value_skipped}")

        # Actually load
        filtered = {k: v for k, v in state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        check(True, "Warm start loaded successfully")
        del ckpt

    # ================================================================
    print("\n" + "=" * 60, flush=True)
    print("5. TRAINING STEP (forward + backward + optimizer)", flush=True)
    print("=" * 60, flush=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Simulate one training step with pairwise data
    obs, pol_tgt, val_tgt, good_obs, bad_obs, margin = ds.collate_pairwise(list(range(16)))

    pol_logits, val_logits = model(obs)

    # Policy loss
    log_probs = F.log_softmax(pol_logits, dim=-1)
    pol_loss = -(pol_tgt * log_probs).sum(dim=-1).mean()

    # Value loss (categorical CE)
    val_loss = -(val_tgt * F.log_softmax(val_logits, dim=-1)).sum(dim=-1).mean()

    # Ranking loss
    pair_obs = torch.cat([good_obs, bad_obs], dim=0)
    _, pair_val = model(pair_obs)
    good_val, bad_val = pair_val.chunk(2, dim=0)
    v_good = model.predict_value(good_val, max_val=100.0)
    v_bad = model.predict_value(bad_val, max_val=100.0)
    margin_scaled = margin * (5.0 / (margin.mean() + 1e-8))
    rank_loss = F.relu(margin_scaled - (v_good - v_bad)).mean()

    loss = pol_loss + 0.01 * val_loss + 1.0 * rank_loss
    check(not torch.isnan(loss), f"Loss is not NaN: {loss.item():.4f}")
    check(not torch.isinf(loss), f"Loss is not Inf: {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    check(True, f"Backward + step OK (loss={loss.item():.4f}, pol={pol_loss.item():.4f}, "
          f"val={val_loss.item():.4f}, rank={rank_loss.item():.4f})")

    # ================================================================
    print("\n" + "=" * 60, flush=True)
    print("6. CHECKPOINT SAVE/LOAD ROUND-TRIP", flush=True)
    print("=" * 60, flush=True)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name

    ckpt = {
        'epoch': 0,
        'model': model.state_dict(),
        'max_score': 100.0,
        'args': {
            'value_bins': 64, 'value_channels': 32,
            'value_hidden': 512, 'value_dropout': 0.3,
            'num_blocks': 10, 'channels': 256,
        },
    }
    torch.save(ckpt, tmp_path)
    check(os.path.exists(tmp_path), f"Checkpoint saved: {os.path.getsize(tmp_path)/1e6:.1f} MB")

    # Load via load_model (the inference path)
    from alphatrain.evaluate import load_model
    loaded_net, loaded_max = load_model(tmp_path, 'cpu')
    check(loaded_max == 100.0, f"Loaded max_score: {loaded_max}")

    # Verify architecture matches
    loaded_params = sum(p.numel() for p in loaded_net.parameters())
    check(loaded_params == n_params, f"Loaded param count matches: {loaded_params:,}")

    # Verify forward pass produces same output
    model.train(False)
    with torch.no_grad():
        orig_pol, orig_val = model(x)
        load_pol, load_val = loaded_net(x)
    check(torch.allclose(orig_pol, load_pol, atol=1e-5),
          f"Policy output matches after reload (max diff: {(orig_pol - load_pol).abs().max():.6f})")
    check(torch.allclose(orig_val, load_val, atol=1e-5),
          f"Value output matches after reload (max diff: {(orig_val - load_val).abs().max():.6f})")

    os.unlink(tmp_path)

    # ================================================================
    print("\n" + "=" * 60, flush=True)
    print("ALL CHECKS PASSED — safe to start Pillar 2k training", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
