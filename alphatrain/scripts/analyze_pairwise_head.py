"""Post-hoc analysis of a trained pairwise SpatialValueHead.

Per ChatGPT review 2026-05-14: aggregate val accuracy can hide useful
subproblems. Break it down by:
  - metric: cap_rate vs turns vs score
  - source: crisis vs selfplay
  - prior rank: did winner have lower or higher policy prior than loser?
    (recomputed by running policy on each anchor)

Also report stable-pair distribution audit on the dataset itself.

Usage:
    python -m alphatrain.scripts.analyze_pairwise_head \\
        --backbone alphatrain/data/pillar2z_epoch_19.pt \\
        --pairwise alphatrain/data/pairwise_v3_<stamp>.pt \\
        --head alphatrain/data/value_head_v3_<stamp>.pt \\
        --device mps
"""

import argparse
import hashlib
import numpy as np
import torch

from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation
from alphatrain.value_head import load_spatial


def board_hash_to_bucket(board_int8, n_buckets=10):
    bs = board_int8.tobytes()
    h = hashlib.md5(bs).digest()
    return int.from_bytes(h[:4], 'little') % n_buckets


def build_obs(board, next_pos, next_col, n_next):
    b = np.asarray(board, dtype=np.int8)
    nr = np.asarray(next_pos[:, 0], dtype=np.intp)
    nc = np.asarray(next_pos[:, 1], dtype=np.intp)
    ncol = np.asarray(next_col, dtype=np.intp)
    return build_observation(b, nr, nc, ncol, int(n_next))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', required=True)
    p.add_argument('--pairwise', required=True)
    p.add_argument('--head', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--val-bucket', type=int, default=0)
    p.add_argument('--n-buckets', type=int, default=10)
    args = p.parse_args()

    device = torch.device(args.device)
    fp16 = (args.device != 'cpu')

    print(f"Loading backbone {args.backbone}...", flush=True)
    net, _ = load_model(args.backbone, device, fp16=fp16, jit_trace=False)
    for pp in net.parameters():
        pp.requires_grad = False

    print(f"Loading head {args.head}...", flush=True)
    head, _ = load_spatial(args.head, device=device)
    head.train(False)

    print(f"Loading pairwise data {args.pairwise}...", flush=True)
    data = torch.load(args.pairwise, weights_only=False)
    N = data['anchor_boards'].shape[0]
    print(f"  pairs: {N:,}", flush=True)

    if 'audit' in data:
        a = data['audit']
        print(f"\n=== DATASET AUDIT ===", flush=True)
        print(f"Candidate pairs: {a['n_candidate_pairs']}", flush=True)
        print(f"Pass FULL: {a['n_pass_full']} "
              f"({100*a['n_pass_full']/max(a['n_candidate_pairs'],1):.1f}%)",
              flush=True)
        print(f"Pass HALF-A: {a['n_pass_a']}, HALF-B: {a['n_pass_b']}, "
              f"BOTH: {a['n_pass_both']}", flush=True)
        if a['n_pass_both']:
            agree = 100 * a['n_agree_winner_given_both'] / a['n_pass_both']
            print(f"Winner agreement | both pass: "
                  f"{a['n_agree_winner_given_both']}/{a['n_pass_both']} "
                  f"({agree:.1f}%)", flush=True)
        print(f"STABLE pairs: {a['n_stable']}", flush=True)
        print(f"By metric (full): {a.get('by_metric_full', {})}", flush=True)
        print(f"By metric (stable): {a.get('by_metric_stable', {})}",
              flush=True)

    anchor_bs = data['anchor_boards'].numpy()
    buckets = np.array([board_hash_to_bucket(anchor_bs[i], args.n_buckets)
                        for i in range(N)])
    val_mask = (buckets == args.val_bucket)
    train_idx = np.nonzero(~val_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    print(f"\nSplit: train {len(train_idx):,}  val {len(val_idx):,} "
          f"(by anchor_board hash mod {args.n_buckets} == {args.val_bucket})",
          flush=True)

    metrics = data['metric_names']
    src_label_map = data.get('source_label_map', {})
    inv_src = {v: k for k, v in src_label_map.items()}
    sources = [inv_src.get(int(s), str(int(s)))
               for s in data['source_labels'].tolist()]

    print(f"\nStable-pair distribution (full dataset):", flush=True)
    print(f"  by metric:", flush=True)
    for m in sorted(set(metrics)):
        n = sum(1 for x in metrics if x == m)
        print(f"    {m}: {n}", flush=True)
    print(f"  by source:", flush=True)
    for s in sorted(set(sources)):
        n = sum(1 for x in sources if x == s)
        print(f"    {s}: {n}", flush=True)

    win_boards = data['win_boards'].numpy()
    win_np = data['win_next_pos'].numpy()
    win_nc = data['win_next_col'].numpy()
    win_nn = data['win_n_next'].numpy()
    lose_boards = data['lose_boards'].numpy()
    lose_np = data['lose_next_pos'].numpy()
    lose_nc = data['lose_next_col'].numpy()
    lose_nn = data['lose_n_next'].numpy()

    def head_score(idxs, n_chunk=256):
        """Return (V_w, V_l) numpy arrays for given pair indices."""
        Vw_all, Vl_all = [], []
        for i in range(0, len(idxs), n_chunk):
            sub = idxs[i:i+n_chunk]
            w_obs = np.stack([build_obs(win_boards[k], win_np[k],
                                        win_nc[k], win_nn[k]) for k in sub])
            l_obs = np.stack([build_obs(lose_boards[k], lose_np[k],
                                        lose_nc[k], lose_nn[k]) for k in sub])
            w_t = torch.from_numpy(w_obs).to(device,
                dtype=torch.float16 if fp16 else torch.float32)
            l_t = torch.from_numpy(l_obs).to(device,
                dtype=torch.float16 if fp16 else torch.float32)
            with torch.inference_mode():
                wf = net.backbone_features(w_t).float()
                lf = net.backbone_features(l_t).float()
                Vw = head(wf).squeeze(-1)
                Vl = head(lf).squeeze(-1)
            Vw_all.append(Vw.cpu().numpy())
            Vl_all.append(Vl.cpu().numpy())
        return np.concatenate(Vw_all), np.concatenate(Vl_all)

    print(f"\n=== HEAD EVAL ===", flush=True)
    Vw_train, Vl_train = head_score(train_idx)
    Vw_val, Vl_val = head_score(val_idx)
    train_correct = (Vw_train > Vl_train)
    val_correct = (Vw_val > Vl_val)

    print(f"Train acc: {train_correct.mean()*100:.1f}% (n={len(train_idx)})",
          flush=True)
    print(f"Val acc:   {val_correct.mean()*100:.1f}% (n={len(val_idx)})",
          flush=True)
    se_val = 100 * np.sqrt(val_correct.mean() * (1 - val_correct.mean()) /
                           max(len(val_idx), 1))
    print(f"Val SE:    ±{se_val:.2f}%", flush=True)

    print(f"\nVal acc by metric:", flush=True)
    val_metrics = [metrics[i] for i in val_idx]
    for m in sorted(set(val_metrics)):
        mask = np.array([x == m for x in val_metrics])
        if mask.sum() == 0:
            continue
        acc = val_correct[mask].mean()
        se = np.sqrt(acc * (1 - acc) / mask.sum())
        print(f"  {m}: n={int(mask.sum())}  acc={acc*100:.1f}% "
              f"±{se*100:.1f}%", flush=True)

    print(f"\nVal acc by source:", flush=True)
    val_sources = [sources[i] for i in val_idx]
    for s in sorted(set(val_sources)):
        mask = np.array([x == s for x in val_sources])
        if mask.sum() == 0:
            continue
        acc = val_correct[mask].mean()
        se = np.sqrt(acc * (1 - acc) / mask.sum())
        print(f"  {s}: n={int(mask.sum())}  acc={acc*100:.1f}% "
              f"±{se*100:.1f}%", flush=True)

    print(f"\nVal acc by metric × source:", flush=True)
    for m in sorted(set(val_metrics)):
        for s in sorted(set(val_sources)):
            mask = np.array([(val_metrics[i] == m and val_sources[i] == s)
                             for i in range(len(val_idx))])
            if mask.sum() < 5:
                continue
            acc = val_correct[mask].mean()
            print(f"  {m} × {s}: n={int(mask.sum())}  acc={acc*100:.1f}%",
                  flush=True)

    Vdiff_val = Vw_val - Vl_val
    print(f"\nV_w - V_l (val): mean={Vdiff_val.mean():.3f}  "
          f"median={np.median(Vdiff_val):.3f}  "
          f"std={Vdiff_val.std():.3f}", flush=True)
    Vdiff_train = Vw_train - Vl_train
    print(f"V_w - V_l (train): mean={Vdiff_train.mean():.3f}  "
          f"std={Vdiff_train.std():.3f}", flush=True)
    print(f"Head output range (val V_w): "
          f"[{Vw_val.min():.3f}, {Vw_val.max():.3f}], "
          f"mean={Vw_val.mean():.3f}", flush=True)
    print(f"Head output range (val V_l): "
          f"[{Vl_val.min():.3f}, {Vl_val.max():.3f}], "
          f"mean={Vl_val.mean():.3f}", flush=True)


if __name__ == '__main__':
    main()
