"""Which corpus trained this checkpoint? Behavioral membership fingerprint.

The pillar3d_mC re-run (19.6k corpus, 2,593 games) overwrote the original run's
(13.8k, 1,837 games) Drive checkpoints — same filenames. The saved args are
identical, so the .pt can't say. But the MODEL remembers its training set:
match rate (argmax == mcts_top) is elevated on corrections it trained on.

Split the corr_*.json files by MINING CHRONOLOGY (mtime) into:
  A = first 1,837 games   (in BOTH candidate corpora)
  B = games 1838..2593    (only in the 19.6k corpus)
  C = games 2594..end     (in NEITHER — clean control)
Validate the split by checking cumulative decisive counts (~13.8k after A,
19,368 after A+B — the known corpus sizes). Then:
  match(A) high, match(B) ~= match(C) low  => the 13.8k model
  match(A) ~= match(B) high, match(C) low  => the 19.6k model
(The 15% by-seed holdout dilutes the trained-on sets slightly; pattern survives.)

    PYTHONPATH=. python scripts/fingerprint_corpus_membership.py \\
        --model alphatrain/data/pillar3d_mC_dec_T05_epoch_2.pt
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation


def collect(files, min_margin=0.05):
    """Decisive corrections from a list of corr files -> (obs[N,18,9,9], top[N])."""
    obs, tops = [], []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for c in d['corrections']:
            if c['mcts_top_share'] - c['pol_share'] < min_margin:
                continue
            board = np.array(c['board'], dtype=np.int8)
            nr = np.zeros(3, dtype=np.int64)
            nc = np.zeros(3, dtype=np.int64)
            ncol = np.zeros(3, dtype=np.int64)
            nb = c['next_balls'][:3]
            for i, ((r, cc), color) in enumerate(nb):
                nr[i], nc[i], ncol[i] = int(r), int(cc), int(color)
            obs.append(build_observation(board, nr, nc, ncol, len(nb)))
            tops.append(int(c['mcts_top_idx']))
    return np.stack(obs), np.array(tops, dtype=np.int64)


@torch.no_grad()
def match_rate(net, dev, obs, tops, bs=1024):
    hits = 0
    for i in range(0, len(obs), bs):
        x = torch.from_numpy(obs[i:i + bs]).to(dev)
        out = net(x)
        logits = (out[0] if isinstance(out, tuple) else out).float()
        hits += (logits.argmax(1).cpu().numpy() == tops[i:i + bs]).sum()
    return hits / len(obs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--corr-dir', default='crisis/corrections')
    p.add_argument('--device', default='mps')
    a = p.parse_args()

    files = glob.glob(os.path.join(a.corr_dir, 'corr_*.json'))
    files.sort(key=lambda f: os.path.getmtime(f))
    print(f"{len(files)} corr files (mtime-ordered)", flush=True)
    splits = {'A (1..1837, in both)': files[:1837],
              'B (1838..2593, only 19.6k)': files[1837:2593],
              'C (2594.., in neither)': files[2593:]}

    dev = torch.device(a.device)
    net, _ = load_model(a.model, dev, fp16=False)
    net.train(False)

    rates = {}
    cum = 0
    for name, fl in splits.items():
        obs, tops = collect(fl)
        cum += len(obs)
        r = match_rate(net, dev, obs, tops)
        rates[name] = r
        print(f"  {name}: {len(obs)} decisive corrections (cum {cum}), "
              f"match={r:.3f}", flush=True)

    rA, rB, rC = rates.values()
    print()
    if rB - rC > 0.5 * (rA - rC):
        print(f"VERDICT: trained on the 19.6k corpus (B elevated like A: "
              f"A={rA:.3f} B={rB:.3f} vs control C={rC:.3f})")
    else:
        print(f"VERDICT: trained on the 13.8k corpus (B at control level: "
              f"A={rA:.3f} B={rB:.3f} C={rC:.3f})")


if __name__ == '__main__':
    main()
