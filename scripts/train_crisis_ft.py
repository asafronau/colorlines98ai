"""Task-arithmetic stage 1: fine-tune the base policy ONLY on crisis corrections.

Gemini-recommended decoupling (Ilharco et al., "Editing Models with Task Arithmetic"):
instead of fighting gradient competition (mC's main+aux dual loss; pillar3e's KL rubber
band), train θ_crisis to fit the corrections WITHOUT any preservation term — let it
forget general play — then restore general play by weight interpolation
(scripts/merge_checkpoints.py): θ_deploy = θ_base + α·(θ_crisis − θ_base), sweeping α
with the cheap local eval.

CRITICAL (our BN-contamination history): BatchNorm runs in eval mode for every forward —
running stats stay EXACTLY the base's (affine params still train), so the task vector
contains no BN-statistics drift and interpolation mixes one normalization regime. The
merge script verifies running stats are unchanged.

Local-friendly: 27.6k corrections ≈ 160MB of obs; ~minutes/epoch on M5 MPS.

    PYTHONPATH=. python scripts/train_crisis_ft.py \\
        --corpus crisis/corrections_corpus_mm05.pt \\
        --base alphatrain/data/pillar3b_epoch_20.pt \\
        --epochs 20 --lr 1e-4 --batch 1024 \\
        --save-dir checkpoints/crisis_ft
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn

from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation
from alphatrain.counterfactual import soft_correction_loss
from alphatrain.train_path_b import frozen_bn
from alphatrain.mcts import _legal_priors_jit


def build_legal_mask(boards, device):
    """Dense bool legal-move mask per state (True = legal)."""
    n = boards.shape[0]
    mask = np.zeros((n, 6561), dtype=bool)
    dummy = np.zeros(6561, dtype=np.float32)
    for i in range(n):
        k, fi, _ = _legal_priors_jit(boards[i], dummy, 6561)
        mask[i, fi[:k]] = True
    return torch.from_numpy(mask).to(device)


def load_corpus(path, device, holdout_frac, split_seed):
    c = torch.load(path, map_location='cpu', weights_only=False)
    n = c['boards'].size(0)
    print(f"corpus: {n} corrections from {c['_stats']['n_seeds']} seeds "
          f"(min_margin={c['_stats']['min_margin']})", flush=True)
    boards = c['boards'].numpy()
    npos = c['next_pos'].numpy().astype(np.int64)
    ncol = c['next_col'].numpy().astype(np.int64)
    nnext = c['n_next'].numpy()
    t0 = time.time()
    obs = np.stack([build_observation(boards[i], npos[i, :, 0], npos[i, :, 1],
                                      ncol[i], int(nnext[i]))
                    for i in range(n)])
    print(f"obs built in {time.time()-t0:.0f}s", flush=True)
    obs = torch.from_numpy(obs).to(device)
    legal = build_legal_mask(boards, device)
    print(f"legal mask built ({legal.sum(1).float().mean():.0f} legal/state avg)",
          flush=True)

    seeds_all = c['seed'].tolist()
    uniq = sorted(set(int(s) for s in seeds_all))
    n_hold = (max(1, int(round(holdout_frac * len(uniq))))
              if holdout_frac > 0 else 0)
    g = torch.Generator().manual_seed(split_seed)
    hperm = torch.randperm(len(uniq), generator=g).tolist()
    hold = set(uniq[i] for i in hperm[:n_hold])
    hold_mask = torch.tensor([int(s) in hold for s in seeds_all])
    tr = (~hold_mask).nonzero(as_tuple=True)[0]
    ho = hold_mask.nonzero(as_tuple=True)[0]
    w_tr = c['weight'][tr]
    w_tr = (w_tr / w_tr.mean().clamp(min=1e-6)).to(device)
    mk = lambda ix: {'obs': obs[ix], 'tgt_idx': c['tgt_idx'][ix].to(device),
                     'tgt_prob': c['tgt_prob'][ix].to(device),
                     'legal': legal[ix],
                     **({'vh1_move': c['vh1_move'][ix].to(device)}
                        if 'vh1_move' in c else {})}
    train, held = mk(tr), (mk(ho) if ho.numel() else None)
    train['weight'] = w_tr
    print(f"train={tr.numel()} heldout={ho.numel()} "
          f"({n_hold}/{len(uniq)} seeds)", flush=True)
    return train, held


@torch.no_grad()
def split_metrics(model, sub, bs=2048):
    model.train(False)
    N = sub['obs'].size(0)
    match, ce, pref = 0, 0.0, 0
    for i in range(0, N, bs):
        out = model(sub['obs'][i:i + bs])
        logits = (out[0] if isinstance(out, tuple) else out).float()
        ti, tp = sub['tgt_idx'][i:i + bs], sub['tgt_prob'][i:i + bs]
        logp = torch.log_softmax(logits, 1)  # CE on raw logits (padded ti rows)
        ce += (-(tp * torch.gather(logp, 1, ti)).sum(1)).sum().item()
        if 'legal' in sub:  # deployment-equivalent: argmax over LEGAL moves
            logits = logits.masked_fill(~sub['legal'][i:i + bs], float('-inf'))
        match += (logits.argmax(1) == ti[:, 0]).sum().item()
        if 'vh1_move' in sub:
            lt = logits.gather(1, ti[:, :1]).squeeze(1)
            ls = logits.gather(1, sub['vh1_move'][i:i + bs].unsqueeze(1)).squeeze(1)
            pref += (lt > ls).sum().item()
    return match / N, ce / N, (pref / N if 'vh1_move' in sub else -1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default='crisis/corrections_corpus_mm05.pt')
    p.add_argument('--base', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0,
                   help='0: weight decay would shrink the task vector toward the '
                        'origin, not toward base — off by default.')
    p.add_argument('--target-temperature', type=float, default=0.5)
    p.add_argument('--loss', choices=['soft', 'margin', 'legalmax'],
                   default='soft',
                   help="margin: pairwise hinge vs vh1's move only. legalmax "
                        "(R2b review arm): relu(m + max_{legal a != teacher} "
                        "logit[a] - logit[teacher]) — directly optimizes legal "
                        "top-1 adoption with dynamic hard negatives, stops "
                        "once the teacher wins by m.")
    p.add_argument('--margin', type=float, default=0.15)
    p.add_argument('--kl-anchor-weight', type=float, default=0.0,
                   help='lambda for CE-to-base anchor on independent quiet '
                        'states (KL(base||model) up to a constant).')
    p.add_argument('--anchor-games-dir', default='data/dagger_games_v1')
    p.add_argument('--anchor-skip-games', type=int, default=200,
                   help='skip the first N games (used by the gate holdout).')
    p.add_argument('--anchor-games', type=int, default=150)
    p.add_argument('--anchor-batch', type=int, default=1024)
    p.add_argument('--shuffle-seed', type=int, default=0)
    p.add_argument('--weighted', action='store_true', default=True)
    p.add_argument('--holdout-frac', type=float, default=0.15)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--device', default='mps')
    p.add_argument('--save-dir', default='checkpoints/crisis_ft')
    p.add_argument('--save-every', type=int, default=5)
    a = p.parse_args()

    dev = torch.device(a.device)
    torch.manual_seed(a.shuffle_seed)
    net, _ = load_model(a.base, dev, fp16=False)
    base_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    train, held = load_corpus(a.corpus, dev, a.holdout_frac, a.split_seed)

    anchor_obs, anchor_probs = None, None
    if a.kl_anchor_weight > 0:
        from alphatrain.scripts.diag_dagger1_regression import build_holdout
        os.makedirs(a.save_dir, exist_ok=True)
        tmp = os.path.join(a.save_dir, 'anchor.tmp')
        n_anchor = build_holdout(a.anchor_games_dir, a.corpus, a.anchor_games,
                                 100, tmp, np.random.default_rng(7),
                                 skip=a.anchor_skip_games)
        t = torch.load(tmp, map_location='cpu', weights_only=True)
        os.remove(tmp)
        ao = np.stack([build_observation(
            t['boards'][i].numpy(), t['next_pos'][i, :, 0].numpy().astype(np.int64),
            t['next_pos'][i, :, 1].numpy().astype(np.int64),
            t['next_col'][i].numpy().astype(np.int64), int(t['n_next'][i]))
            for i in range(n_anchor)])
        anchor_obs = torch.from_numpy(ao).to(dev)
        with torch.no_grad():
            ps = []
            for i in range(0, n_anchor, 2048):
                out = net(anchor_obs[i:i + 2048])
                lg = (out[0] if isinstance(out, tuple) else out).float()
                ps.append(torch.softmax(lg, 1).half())
            anchor_probs = torch.cat(ps)
        print(f"anchor: {n_anchor:,} quiet states from games "
              f"[{a.anchor_skip_games}:{a.anchor_skip_games + a.anchor_games}], "
              f"lambda={a.kl_anchor_weight}", flush=True)
    N = train['obs'].size(0)
    steps = (N + a.batch - 1) // a.batch
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    os.makedirs(a.save_dir, exist_ok=True)

    mr, ce, pf = split_metrics(net, train)
    line = f"[ep0] train match={mr:.3f} ce={ce:.3f} pref={pf:.3f}"
    if held:
        mh, ch, ph = split_metrics(net, held)
        line += f" | HELD match={mh:.3f} ce={ch:.3f} pref={ph:.3f}"
    print(line, flush=True)

    for ep in range(1, a.epochs + 1):
        net.train(True)
        perm = torch.randperm(N, device=dev)
        tot = 0.0
        t0 = time.time()
        for b in range(steps):
            idx = perm[b * a.batch:(b + 1) * a.batch]
            # frozen_bn: BN normalizes with the BASE's running stats and never
            # updates them — the whole point of the task-vector approach here.
            with frozen_bn(net):
                out = net(train['obs'][idx])
            logits = out[0] if isinstance(out, tuple) else out
            if a.loss == 'margin':
                lt = logits.gather(1, train['tgt_idx'][idx][:, :1]).squeeze(1)
                ls = logits.gather(
                    1, train['vh1_move'][idx].unsqueeze(1)).squeeze(1)
                loss = torch.relu(a.margin - (lt - ls)).mean()
            elif a.loss == 'legalmax':
                t1 = train['tgt_idx'][idx][:, :1]
                lg = logits.masked_fill(~train['legal'][idx], float('-inf'))
                lt = lg.gather(1, t1).squeeze(1)
                hard_neg = lg.scatter(1, t1, float('-inf')).max(1).values
                loss = torch.relu(a.margin + hard_neg - lt).mean()
            else:
                loss = soft_correction_loss(
                    logits, train['tgt_idx'][idx], train['tgt_prob'][idx],
                    anchor_weight=(train['weight'][idx] if a.weighted else None),
                    target_temperature=a.target_temperature)
            if anchor_probs is not None:
                aidx = torch.randint(0, anchor_probs.shape[0],
                                     (a.anchor_batch,), device=dev)
                with frozen_bn(net):
                    aout = net(anchor_obs[aidx])
                alg = aout[0] if isinstance(aout, tuple) else aout
                ce_a = -(anchor_probs[aidx].float()
                         * torch.log_softmax(alg.float(), 1)).sum(1).mean()
                loss = loss + a.kl_anchor_weight * ce_a
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
        mr, ce, pf = split_metrics(net, train)
        line = (f"[ep{ep}] loss={tot/steps:.4f} train match={mr:.3f} "
                f"ce={ce:.3f} pref={pf:.3f}")
        if held:
            mh, ch, ph = split_metrics(net, held)
            line += f" | HELD match={mh:.3f} ce={ch:.3f} pref={ph:.3f}"
        print(line + f"  [{time.time()-t0:.0f}s]", flush=True)

        if ep % a.save_every == 0 or ep == a.epochs:
            # Sanity: BN running stats must be bit-identical to base.
            sd = net.state_dict()
            for k in sd:
                if 'running_mean' in k or 'running_var' in k:
                    assert torch.equal(sd[k], base_state[k]), \
                        f"BN stat drifted: {k} — frozen_bn failed"
            path = os.path.join(a.save_dir, f'ft_epoch_{ep}.pt')
            torch.save({'model': sd, 'epoch': ep, 'args': vars(a),
                        'policy_only': True, 'base': a.base}, path)
            print(f"  saved {path} (BN stats verified == base)", flush=True)


if __name__ == '__main__':
    main()
