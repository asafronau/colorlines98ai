"""Probe: is the merge failure OVERFIT-NOISE (Gemini) or EDIT-CHANNEL-TOO-BROAD (ChatGPT)?

Both reviewers converge: the full-network task-arith merge causes BROAD drift on normal play
that swamps the small crisis gains. They split on cause: Gemini = overfit noise (too few
games -> noisy vector -> more data fixes); ChatGPT = the edit channel is too broad (restrict
the merge; more data won't help). This probe resolves it WITHOUT re-mining, using the EXISTING
ft vector.

Merge the existing vector RESTRICTED (policy-head-only, last-block+head) vs FULL, across small
alpha. Measure OFFLINE (cheap forwards, no gameplay):
  LOCAL MATCH  = on correction states, does the merged model's legal-argmax == MCTS-top?
                 (does the merge actually INSTALL the corrections?)
  BROAD DRIFT  = on RANDOM pillar3f self-play states (sampled from v15, the real normal-play
                 distribution), fraction where merged legal-argmax != base legal-argmax.
                 (does the merge DAMAGE normal play?)

A good edit: local-match UP, broad-drift LOW. Read:
  head-only gives high local-match + low broad-drift  -> edit channel too broad (ChatGPT);
       fix = restricted merge / LoRA; more data alone would NOT have fixed it.
  ALL scopes keep broad-drift high                     -> overfit noise (Gemini); more data / LoRA.
  local-match never improves                           -> distillation target/training failed.

    PYTHONPATH=. python scripts/probe_merge_locality.py
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.model import AlphaTrainNet
from alphatrain.observation import build_observation
from alphatrain.mcts import _get_legal_priors_flat

FROZEN = ('running_mean', 'running_var', 'num_batches_tracked')  # frozen-BN: no-op to merge


def state_of(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    st = ck['model']
    if any(k.startswith('_orig_mod.') for k in st):
        st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    return st


def in_scope(k, scope):
    if any(f in k for f in FROZEN):
        return False
    if scope == 'full':
        return True
    if scope == 'head':
        return k.startswith(('policy_conv1', 'policy_bn', 'policy_conv2'))
    if scope == 'lastblock_head':
        return k.startswith(('policy_conv1', 'policy_bn', 'policy_conv2', 'blocks.9.'))
    return False


def merged_state(base, ft, scope, alpha):
    out = {}
    for k in base:
        if in_scope(k, scope) and k in ft and base[k].dtype.is_floating_point:
            out[k] = base[k] + alpha * (ft[k] - base[k])
        else:
            out[k] = base[k]
    return out


def legal_argmax_batch(net, dev, dtype, boards, npos, ncol, nnext):
    """Per-state legal argmax move (flat idx). Returns list."""
    obs = np.stack([build_observation(boards[i], npos[i, :, 0], npos[i, :, 1],
                                      ncol[i], int(nnext[i])) for i in range(len(boards))])
    out = []
    for s in range(0, len(boards), 1024):
        ob = torch.from_numpy(obs[s:s+1024]).to(dev, dtype)
        with torch.no_grad():
            o = net(ob)
            lg = (o[0] if isinstance(o, tuple) else o).float().cpu().numpy()
        for i in range(lg.shape[0]):
            pri = _get_legal_priors_flat(boards[s+i], lg[i], 30)
            out.append(max(pri.items(), key=lambda x: x[1])[0] if pri else -1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='alphatrain/data/pillar3f.pt')
    p.add_argument('--ft', default='checkpoints/crisis_ft_pillar3f_mcts/ft_epoch_15.pt')
    p.add_argument('--corr-glob', default='crisis/corrections_pillar3f/corr_*.json')
    p.add_argument('--broad-tensor', default='alphatrain/data/v15_pillar3f_slim.pt')
    p.add_argument('--n-local', type=int, default=400)
    p.add_argument('--n-broad', type=int, default=800)
    p.add_argument('--scopes', default='head,lastblock_head,full')
    p.add_argument('--alphas', default='0.1,0.2,0.4')
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)
    rng = np.random.default_rng(0)

    base, ft = state_of(a.base), state_of(a.ft)

    # ---- local sample: correction states (board, next_balls, mcts_top, pol) ----
    L = []
    for f in sorted(glob.glob(a.corr_glob)):
        try: d = json.load(open(f))
        except Exception: continue
        for c in d['corrections']:
            if c['mcts_top_idx'] != c['pol_idx']:
                L.append(c)
        if len(L) >= a.n_local * 4: break
    li = rng.choice(len(L), size=min(a.n_local, len(L)), replace=False)
    L = [L[i] for i in li]
    def parse_nb(nbs):  # nbs = [[[r,c], color], ...]; pad to 3
        nbs = (list(nbs) + [[[0, 0], 0]] * 3)[:3]
        return ([[int(nb[0][0]), int(nb[0][1])] for nb in nbs],
                [int(nb[1]) for nb in nbs])
    Lb = [np.array(c['board'], dtype=np.int8) for c in L]
    _pos = [parse_nb(c['next_balls']) for c in L]
    Lnp = np.array([p for p, _ in _pos]); Lnc = np.array([col for _, col in _pos])
    Lnn = np.array([min(len(c['next_balls']), 3) for c in L])
    L_mcts = [int(c['mcts_top_idx']) for c in L]

    # ---- broad sample: real pillar3f self-play states from v15 ----
    d = torch.load(a.broad_tensor, map_location='cpu', mmap=True, weights_only=False)
    nt = d['boards'].shape[0]
    bi = np.sort(rng.choice(nt, size=a.n_broad, replace=False))
    Bb = d['boards'][bi].numpy().astype(np.int8)
    Bnp = d['next_pos'][bi].numpy().astype(np.int64)
    Bnc = d['next_col'][bi].numpy().astype(np.int64)
    Bnn = d['n_next'][bi].numpy().astype(np.int64)

    def build(state):
        m = AlphaTrainNet(num_blocks=10, channels=256).to(dev)
        ms = m.state_dict()
        m.load_state_dict({k: v for k, v in state.items() if k in ms}, strict=False)
        m.train(False)
        return m.half() if dev.type in ('mps', 'cuda') else m

    dtype = torch.float16 if dev.type in ('mps', 'cuda') else torch.float32
    base_net = build(base)
    base_local = legal_argmax_batch(base_net, dev, dtype, Lb, Lnp, Lnc, Lnn)
    base_broad = legal_argmax_batch(base_net, dev, dtype, Bb, Bnp, Bnc, Bnn)
    base_local_match = np.mean([base_local[i] == L_mcts[i] for i in range(len(L))])
    print(f"BASE pillar3f: local-match-to-MCTS-top {100*base_local_match:.1f}% "
          f"(this is what the merge must IMPROVE)\n", flush=True)

    print(f"{'scope':>16} {'alpha':>6} {'local-match%':>12} {'(vs base)':>10} "
          f"{'broad-drift%':>12}", flush=True)
    print('-'*62, flush=True)
    for scope in a.scopes.split(','):
        for alpha in [float(x) for x in a.alphas.split(',')]:
            net = build(merged_state(base, ft, scope, alpha))
            lm = legal_argmax_batch(net, dev, dtype, Lb, Lnp, Lnc, Lnn)
            bm = legal_argmax_batch(net, dev, dtype, Bb, Bnp, Bnc, Bnn)
            local_match = np.mean([lm[i] == L_mcts[i] for i in range(len(L))])
            broad_drift = np.mean([bm[i] != base_broad[i] for i in range(len(Bb))])
            print(f"{scope:>16} {alpha:>6} {100*local_match:>11.1f}% "
                  f"{100*(local_match-base_local_match):>+9.1f}% {100*broad_drift:>11.1f}%",
                  flush=True)
    print("\n--- READ ---", flush=True)
    print("Want: local-match UP (installs corrections) + broad-drift LOW (spares normal play).")
    print("head-only high-local/low-broad => edit channel too broad (ChatGPT) -> restricted merge/LoRA.")
    print("all scopes high broad-drift => overfit noise (Gemini) -> more data / LoRA.")
    print("local-match never rises => distillation target failed.")


if __name__ == '__main__':
    main()
