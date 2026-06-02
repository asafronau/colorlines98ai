"""Fork-count learning curve: does held-out crisis-fork generalization scale
with the NUMBER of training forks?

For each K, warm-start from pillar3b and fine-tune with the SAME recipe as the
real run (sharpened V12 distillation anchor + confidence-weighted listwise aux,
BN frozen during the aux forward), but using only K of the train forks. The
held-out fork set (59 forks from disjoint SEEDS) is FIXED across all K. Because
the aux overfits, we track the held-out flip/margin trajectory and report the
PEAK (early-stop point) per K.

Reads the curve:
  rising at K=233  -> more forks help; slope predicts how many to mine
  flat by K~150    -> forks aren't the lever (low ceiling)

    PYTHONPATH=. python scripts/fork_learning_curve.py
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.model import AlphaTrainNet
from alphatrain.train_path_b import distillation_loss, frozen_bn
from alphatrain.counterfactual import (build_crisis_corpus, listwise_margin_loss,
                                       preflight_metrics)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', default='alphatrain/data/v13_pillar3a.pt')
    p.add_argument('--ckpt', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--ks', default='50,100,150,200,233')
    p.add_argument('--steps', type=int, default=180)
    p.add_argument('--eval-every', type=int, default=15)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lam', type=float, default=0.15)
    p.add_argument('--margin', type=float, default=0.25)
    p.add_argument('--target-temp', type=float, default=0.5)
    p.add_argument('--main-batch', type=int, default=4096)
    p.add_argument('--aux-batch', type=int, default=128)
    p.add_argument('--holdout-frac', type=float, default=0.2)
    p.add_argument('--split-seed', type=int, default=0)
    a = p.parse_args()

    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={dev}", flush=True)

    # Main corpus (augment off -> the only variable is K). collate() gives
    # (obs, policy_target) for random sample indices.
    train_set, _ = TensorDatasetGPU.make_train_val_split(
        a.tensor, val_split=0.01, augment=False, color_augment=False,
        augment_factor=1, device=str(dev), seed=42)
    n_main = len(train_set)

    # Crisis corpus + canonical obs (same builder as the real aux path)
    corpus = build_crisis_corpus('logs/mine_*.json', device='cpu')
    with torch.no_grad():
        fork_obs = train_set._build_obs_core(
            corpus['boards'].long().to(dev), next_pos=corpus['next_pos'].to(dev),
            next_col=corpus['next_col'].to(dev), n_next=corpus['n_next'].to(dev))
    win = corpus['winner_idx'].to(dev); top1 = corpus['top1_idx'].to(dev)
    los = corpus['loser_idx'].to(dev); lmask = corpus['loser_mask'].to(dev)
    weight = corpus['weight'].to(dev); seeds = corpus['seed'].tolist()

    # Fixed by-seed split (identical to train_path_b)
    uniq = sorted(set(int(s) for s in seeds))
    g = np.random.default_rng(a.split_seed)
    perm = g.permutation(len(uniq))
    n_hold = max(1, int(round(a.holdout_frac * len(uniq))))
    hold_seeds = set(uniq[i] for i in perm[:n_hold])
    hold_idx = np.array([i for i, s in enumerate(seeds) if int(s) in hold_seeds])
    train_idx = np.array([i for i, s in enumerate(seeds) if int(s) not in hold_seeds])
    print(f"forks: {len(train_idx)} train / {len(hold_idx)} heldout "
          f"({n_hold}/{len(uniq)} seeds)\n", flush=True)

    hold = dict(obs=fork_obs[torch.tensor(hold_idx, device=dev)],
                win=win[torch.tensor(hold_idx, device=dev)],
                top1=top1[torch.tensor(hold_idx, device=dev)],
                los=los[torch.tensor(hold_idx, device=dev)],
                mask=lmask[torch.tensor(hold_idx, device=dev)])

    @torch.no_grad()
    def held_metrics(model):
        model.train(False)
        logits = model(hold['obs']).float()
        return preflight_metrics(logits, hold['win'], hold['top1'],
                                 hold['los'], hold['mask'], margin=a.margin)

    base_state = {k: v.clone() for k, v in
                  AlphaTrainNet(num_blocks=10, channels=256).state_dict().items()}
    raw = torch.load(a.ckpt, map_location=dev, weights_only=False)['model']
    if any(k.startswith('_orig_mod.') for k in raw):
        raw = {k.replace('_orig_mod.', ''): v for k, v in raw.items()}
    pillar3b_state = {k: v for k, v in raw.items()
                      if k in base_state and v.shape == base_state[k].shape}

    ks = [int(x) for x in a.ks.split(',')]
    rng = np.random.default_rng(0)
    print(f"{'K':>5} {'peakFlip':>9} {'margin@peak':>11} {'conc@peak':>10} "
          f"{'@step':>6}  (baseline flip=0, margin=-2.99)", flush=True)
    print('-' * 60, flush=True)
    results = []
    for K in ks:
        model = AlphaTrainNet(num_blocks=10, channels=256).to(dev)
        model.load_state_dict(pillar3b_state, strict=False)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        kt = rng.choice(train_idx, min(K, len(train_idx)), replace=False)
        kt_t = torch.tensor(kt, device=dev)
        w_k = weight[kt_t]; w_k = w_k / w_k.mean().clamp(min=1e-6)
        peak = {'flip': -1.0, 'margin': None, 'conc': None, 'step': -1}
        ptr = 0
        aperm = torch.randperm(len(kt_t), device=dev)
        for step in range(a.steps):
            model.train(True)
            mi = torch.randint(0, n_main, (a.main_batch,)).tolist()
            obs, tgt = train_set.collate(mi)
            main_loss = distillation_loss(model(obs), tgt,
                                          target_temperature=a.target_temp)
            if ptr + a.aux_batch > len(kt_t):
                aperm = torch.randperm(len(kt_t), device=dev); ptr = 0
            sel = aperm[ptr:ptr + a.aux_batch]; ptr += a.aux_batch
            ai = kt_t[sel]
            with frozen_bn(model):
                alog = model(fork_obs[ai])
            aux_loss = listwise_margin_loss(alog, win[ai], top1[ai], los[ai],
                                            lmask[ai], margin=a.margin,
                                            anchor_weight=w_k[sel])
            loss = main_loss + a.lam * aux_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % a.eval_every == 0 or step == a.steps - 1:
                m = held_metrics(model)
                if m['stored_top1_flip_rate'] > peak['flip']:
                    peak = {'flip': m['stored_top1_flip_rate'],
                            'margin': m['mean_top1_margin'],
                            'conc': m['clean_loser_concordance'], 'step': step}
        results.append((K, peak))
        print(f"{K:>5} {peak['flip']:>9.3f} {peak['margin']:>11.2f} "
              f"{peak['conc']:>10.3f} {peak['step']:>6}", flush=True)

    print("\nK vs peak held-out flip:", flush=True)
    for K, pk in results:
        bar = '#' * int(pk['flip'] * 100)
        print(f"  K={K:>4}: {pk['flip']:.3f} {bar}", flush=True)


if __name__ == '__main__':
    main()
