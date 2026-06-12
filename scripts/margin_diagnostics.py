"""Encoding-depth (margin) diagnostics across the corpus-size ladder.

H1-dilution test without GPU spend (ChatGPT-suggested metrics): all models match
~equally on never-seen corrections (argmax saturates), but if per-correction weight
halves with corpus size, the ENCODING should weaken — lower teacher-top1 prob, lower
top-K mass, smaller logit margins, higher soft CE. Compare across:
base -> ctrl0 -> mC 13.8k -> 19.6k -> 27.6k (+ mF/mG for the λ axis).

Common test set: corrections from games mined AFTER the 19.6k cut (mtime slice C/D,
file index >= 2593) INTERSECTED with the 27.6k trainer's by-seed holdout (replicated
exactly: sorted unique seeds, torch.Generator(0), 15%) — unseen by EVERY model.

    PYTHONPATH=. python scripts/margin_diagnostics.py
"""
import os, sys, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation

CORPUS = 'crisis/corrections_corpus_mm05.pt'
CORR_DIR = 'crisis/corrections'
CUT = 2593          # mtime index: files >= CUT are slice C/D (post-19.6k mining)
HOLDOUT_FRAC, SPLIT_SEED = 0.15, 0
MODELS = [
    ('pillar3b base',  'alphatrain/data/pillar3b_epoch_20.pt'),
    ('ctrl0 (lam=0)',  'alphatrain/data/pillar3d_ctrl0_epoch_2.pt'),
    ('mC 13.8k',       'alphatrain/data/pillar3d_mC_dec_T05_epoch_2.pt'),
    ('mC 19.6k',       'alphatrain/data/pillar3d_mC_dec_T05_2500_epoch_2.pt'),
    ('mC 27.6k',       'alphatrain/data/pillar3d_mC27k_epoch_2.pt'),
    ('mF 19.6k l.012', 'alphatrain/data/pillar3d_mF_dec_lam0.012_epoch_2.pt'),
    ('mG 19.6k l.014', 'alphatrain/data/pillar3d_mG_dec_lam0.014_epoch_2.pt'),
]


def common_unseen_rows(corpus):
    seeds_all = corpus['seed'].tolist()
    uniq = sorted(set(int(s) for s in seeds_all))
    n_hold = max(1, int(round(HOLDOUT_FRAC * len(uniq))))
    g = torch.Generator().manual_seed(SPLIT_SEED)
    hperm = torch.randperm(len(uniq), generator=g).tolist()
    hold_seeds = set(uniq[i] for i in hperm[:n_hold])

    files = glob.glob(os.path.join(CORR_DIR, 'corr_*.json'))
    files.sort(key=lambda f: os.path.getmtime(f))
    cd_seeds = set(int(re.search(r'_(\d+)\.json$', f).group(1))
                   for f in files[CUT:])
    keep = hold_seeds & cd_seeds
    rows = [i for i, s in enumerate(seeds_all) if int(s) in keep]
    print(f"common-unseen: {len(rows)} corrections from {len(keep)} games "
          f"(27.6k-holdout ∩ slice C/D)", flush=True)
    return rows


@torch.no_grad()
def metrics(net, dev, obs_t, tgt_idx, tgt_prob, base_logp=None, bs=1024):
    N = obs_t.size(0)
    match = p1 = massk = ce = marg = kl = 0.0
    for i in range(0, N, bs):
        out = net(obs_t[i:i + bs].to(dev))
        logits = (out[0] if isinstance(out, tuple) else out).float()
        ti = tgt_idx[i:i + bs].to(dev)
        tp = tgt_prob[i:i + bs].to(dev)
        t1 = ti[:, 0]
        logp = torch.log_softmax(logits, 1)
        probs = logp.exp()
        match += (logits.argmax(1) == t1).sum().item()
        p1 += probs.gather(1, t1[:, None]).sum().item()
        massk += (probs.gather(1, ti) * (tp > 0)).sum().item()
        ce += (-(tp * torch.gather(logp, 1, ti)).sum(1)).sum().item()
        l1 = logits.gather(1, t1[:, None]).squeeze(1)
        masked = logits.scatter(1, t1[:, None], float('-inf'))
        marg += (l1 - masked.max(1).values).sum().item()
        if base_logp is not None:
            bl = base_logp[i:i + bs].to(dev)
            kl += (probs * (logp - bl)).sum().item()
    r = dict(match=match / N, p_top1=p1 / N, mass_top20=massk / N,
             softCE=ce / N, margin=marg / N)
    if base_logp is not None:
        r['KL_vs_base'] = kl / N
    return r


def main():
    dev = torch.device('mps')
    corpus = torch.load(CORPUS, map_location='cpu', weights_only=False)
    rows = common_unseen_rows(corpus)
    boards = corpus['boards'][rows].numpy()
    npos = corpus['next_pos'][rows].numpy().astype(np.int64)
    ncol = corpus['next_col'][rows].numpy().astype(np.int64)
    nnext = corpus['n_next'][rows].numpy()
    tgt_idx = corpus['tgt_idx'][rows]
    tgt_prob = corpus['tgt_prob'][rows]
    obs = np.stack([build_observation(boards[i], npos[i, :, 0], npos[i, :, 1],
                                      ncol[i], int(nnext[i]))
                    for i in range(len(rows))])
    obs_t = torch.from_numpy(obs)

    # Base log-probs once (for KL_vs_base)
    base_net, _ = load_model(MODELS[0][1], dev, fp16=False)
    base_net.train(False)
    base_logp = []
    with torch.no_grad():
        for i in range(0, obs_t.size(0), 1024):
            out = base_net(obs_t[i:i + 1024].to(dev))
            logits = (out[0] if isinstance(out, tuple) else out).float()
            base_logp.append(torch.log_softmax(logits, 1).cpu())
    base_logp = torch.cat(base_logp)

    hdr = (f"{'model':<17} {'match':>6} {'p(top1)':>8} {'massK':>6} "
           f"{'softCE':>7} {'margin':>7} {'KLvsBase':>8}")
    print('\n' + hdr)
    print('-' * len(hdr))
    for name, path in MODELS:
        if not os.path.exists(path):
            print(f"{name:<17} MISSING {path}")
            continue
        net, _ = load_model(path, dev, fp16=False)
        net.train(False)
        m = metrics(net, dev, obs_t, tgt_idx, tgt_prob, base_logp)
        print(f"{name:<17} {m['match']:>6.3f} {m['p_top1']:>8.4f} "
              f"{m['mass_top20']:>6.3f} {m['softCE']:>7.3f} {m['margin']:>7.2f} "
              f"{m.get('KL_vs_base', 0):>8.4f}", flush=True)


if __name__ == '__main__':
    main()
