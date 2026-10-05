"""Evaluate survival value heads on within-fullness-bin death-prediction ROC AUC
(reproducing the HISTORY 265/269/270/275 diagnostic).

Evaluates on the held-out validation games (~is_train) of a value_targets tensor
(default: alphatrain/data/value_targets_A8.pt) so models trained on value_targets_A8.pt
or value_targets_pooled_A3_A7.pt are judged strictly out-of-sample.
"""
import argparse
import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model
from alphatrain.value_head import SURVIVAL_HORIZONS, load_any


BINS = [(36, 41), (41, 46), (46, 51), (51, 56), (56, 63)]


def roc_auc(y_true_die, score_die):
    """Exact Mann-Whitney U ROC AUC for binary y_true_die in {0, 1}."""
    pos = score_die[y_true_die == 1]
    neg = score_die[y_true_die == 0]
    if len(pos) < 10 or len(neg) < 10:
        return float('nan')
    order = np.argsort(score_die)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score_die) + 1, dtype=np.float64)
    # Tie correction
    sorted_s = score_die[order]
    i = 0
    n = len(sorted_s)
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        if j > i + 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    r_pos = ranks[y_true_die == 1].sum()
    n_pos, n_neg = float(len(pos)), float(len(neg))
    u = r_pos - n_pos * (n_pos + 1.0) / 2.0
    return float(u / (n_pos * n_neg))


def eval_head(backbone_path, head_path, boards, npos, ncol, nn_arr, labels, masks, balls, device):
    head, ckpt, _ = load_any(head_path, device=device)
    # Support heads that saved their own fine-tuned/unfrozen backbone state_dict
    if 'backbone_state_dict' in ckpt and ckpt['backbone_state_dict'] is not None:
        net, _ = load_model(backbone_path, device, fp16=False, jit_trace=False)
        net.load_state_dict(ckpt['backbone_state_dict'])
        if device.type != 'cpu':
            from alphatrain.inference_cpp.export_ts import fp16_safe_batchnorm
            fp16_safe_batchnorm(net)
            net = net.half()
        net.train(False)
    else:
        net, _ = load_model(backbone_path, device, fp16=(device.type != 'cpu'), jit_trace=False)
        net.train(False)
    head.train(False)
    cat_obs = bool(ckpt.get('cat_obs', False))

    ds_helper = TensorDatasetGPU.__new__(TensorDatasetGPU)
    ds_helper.device = device
    n = len(boards)
    probs = np.zeros((n, len(SURVIVAL_HORIZONS)), dtype=np.float32)
    bs = 2048
    dtype = torch.float16 if device.type != 'cpu' else torch.float32
    with torch.inference_mode():
        for i in range(0, n, bs):
            b_sl = boards[i:i + bs].to(device)
            np_sl = npos[i:i + bs].to(device)
            nc_sl = ncol[i:i + bs].to(device)
            nn_sl = nn_arr[i:i + bs].to(device)
            obs = ds_helper._build_obs_core(b_sl, next_pos=np_sl, next_col=nc_sl, n_next=nn_sl).to(dtype)
            feats = net.backbone_features(obs).float()
            if cat_obs:
                feats = torch.cat([feats, obs.float()], dim=1)
            logits = head(feats)
            probs[i:i + bs] = torch.sigmoid(logits).cpu().numpy()

    print(f"\n=== {head_path} (inner_val={ckpt.get('val_metrics', {}).get('inner_val_loss', float('nan')):.4f}) ===")
    print("   balls   n die/surv@200  AUC H=25   AUC H=50  AUC H=100  AUC H=200   mean P(surv 200) die vs surv")
    aucs_200 = []
    for lo, hi in BINS:
        in_bin = (balls >= lo) & (balls < hi)
        aucs = []
        for hi_idx in range(len(SURVIVAL_HORIZONS)):
            valid = in_bin & (masks[:, hi_idx] > 0)
            y_die = (labels[valid, hi_idx] == 0).astype(np.int32)
            s_die = 1.0 - probs[valid, hi_idx]
            aucs.append(roc_auc(y_die, s_die))
        valid200 = in_bin & (masks[:, 3] > 0)
        die200 = valid200 & (labels[:, 3] == 0)
        surv200 = valid200 & (labels[:, 3] == 1)
        n_die = int(die200.sum())
        n_surv = int(surv200.sum())
        m_die = float(probs[die200, 3].mean()) if n_die > 0 else float('nan')
        m_surv = float(probs[surv200, 3].mean()) if n_surv > 0 else float('nan')
        aucs_200.append(aucs[3])
        print(f"[{lo},{hi})   {n_die:5d}/{n_surv:6d}   "
              f"{aucs[0]:8.3f}   {aucs[1]:8.3f}   {aucs[2]:8.3f}   {aucs[3]:8.3f}   "
              f"{m_die:.3f} vs {m_surv:.3f}")
    print(f"Mean H=200 AUC across bins: {np.nanmean(aucs_200):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='alphatrain/data/value_targets_A8.pt')
    p.add_argument('--backbone', default='alphatrain/data/ta_A7_e4_a1.0.pt')
    p.add_argument('--heads', nargs='+', required=True)
    p.add_argument('--val-only', action='store_true', default=True,
                   help='Evaluate strictly on held-out (~is_train) games (default True)')
    p.add_argument('--all-states', action='store_true',
                   help='Evaluate on a 60k subsample of all games (36-62 balls)')
    p.add_argument('--device', default='mps')
    a = p.parse_args()

    d = torch.load(a.data, weights_only=False)
    boards = d['boards']
    balls = (boards > 0).sum(dim=(1, 2)).numpy()
    sel = (balls >= 36) & (balls < 63)
    if not a.all_states:
        sel = sel & (~d['is_train'].numpy())
    idxs = np.nonzero(sel)[0]
    if a.all_states and len(idxs) > 60000:
        rng = np.random.default_rng(0)
        idxs = np.sort(rng.choice(idxs, size=60000, replace=False))
    print(f"Selected {len(idxs):,} states (36-62 balls, val_only={not a.all_states}) from {a.data}")

    boards_s = boards[idxs]
    npos_s = d['next_pos'][idxs]
    ncol_s = d['next_col'][idxs]
    nn_s = d['n_next'][idxs]
    labels_s = d['survive_labels'][idxs].numpy()
    masks_s = d['survive_masks'][idxs].numpy()
    balls_s = balls[idxs]

    device = torch.device(a.device)
    for h in a.heads:
        eval_head(a.backbone, h, boards_s, npos_s, ncol_s, nn_s, labels_s, masks_s, balls_s, device)


if __name__ == '__main__':
    main()
