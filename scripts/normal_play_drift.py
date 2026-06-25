"""Normal-play preservation guard: did a crisis correction make the model FORGET normal play?

Crisis corrections must be TARGETED — change the policy at crisis / crisis-onset, leave normal
play untouched. (Full-distribution distillation does the opposite: it de-peaks the WHOLE policy
= forgetting, quantified in project_selfplay_gumbel_recipe.) This compares base vs corrected on
positions spanning the difficulty spectrum and reports DRIFT, binned by turns-to-death (a clean,
model-free regime axis — NOT board occupancy, which the user rightly rejected as a metric):

  - HEALTHY (far from death): drift MUST be ~0 — proof normal play was preserved.
  - CRISIS (near death):       drift is the INTENDED fix and should concentrate here.

Drift metrics per position (over the shared legal-move support):
  top1_agree  = base and corrected pick the SAME move (the gameplay-relevant one)
  top3_overlap= overlap of the two top-3 sets
  js          = Jensen-Shannon divergence of the legal-move distributions (0=identical)

A good targeted correction: top1_agree ~1.0 and js ~0 in the HEALTHY bin. If healthy-bin
top1_agree drops / js rises, the correction BLED into normal play -> rein it in (smaller
task-arith alpha, more rehearsal) BEFORE shipping. Pair with full-game eval (mean+median+
ceiling must hold, not just the floor).

    PYTHONPATH=. python scripts/normal_play_drift.py \
        --base alphatrain/data/pillar3b_epoch_20.pt \
        --corrected <crisis_corrected>.pt --n 400
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.evaluate import load_model
from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
from scripts.batched_rollout import restore

BINS = [(1500, 10**9, 'healthy (>1500 to death)'),
        (300, 1500, 'loaded (300-1500)'),
        (0, 300, 'crisis (<300 to death)')]


def regime(ttd):
    for lo, hi, label in BINS:
        if lo <= ttd < hi:
            return label
    return BINS[-1][2]


def js_div(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def sample_positions(games_dir, n, rng):
    """Broad sample spanning regimes: keep each frame's turns-to-death."""
    files = sorted(glob.glob(os.path.join(games_dir, 'death_*.json')))
    rng.shuffle(files)
    out = []
    for f in files:
        if len(out) >= n:
            break
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ft, frames = d.get('final_turns', 0), d.get('frames', [])
        if not frames:
            continue
        # take a few frames spread across this game (healthy..crisis)
        idxs = sorted(set(int(x) for x in np.linspace(5, len(frames) - 1, 6)))
        for ti in idxs:
            fr = frames[ti]
            anc = {'board': fr['board'], 'next_balls': fr['next_balls'],
                   'score': fr.get('score_before', fr['score']), 'turn': fr['turn']}
            out.append({'anchor': anc, 'ttd': ft - fr['turn']})
            if len(out) >= n:
                break
    return out


def model_dist(net, dev, dtype, g, legal):
    obs = torch.from_numpy(_build_obs_for_game(g)).unsqueeze(0).to(dev, dtype)
    with torch.no_grad():
        lg = net(obs)[0].float().cpu().numpy()
    z = lg[legal] - lg[legal].max()
    p = np.exp(z); p /= p.sum()
    return p, lg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--corrected', required=True)
    p.add_argument('--games-dir', default='alphatrain/data/death_games')
    p.add_argument('--n', type=int, default=400)
    p.add_argument('--device', default=None)
    p.add_argument('--out', default='/tmp/normal_play_drift.json')
    a = p.parse_args()

    dev = torch.device(a.device) if a.device else torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available() else 'cpu')
    base, _ = load_model(a.base, dev, fp16=False)
    corr, _ = load_model(a.corrected, dev, fp16=False)
    dtype = next(base.parameters()).dtype
    print(f"Device {dev} | base {a.base}\n            corrected {a.corrected}", flush=True)

    rng = np.random.default_rng(0)
    positions = sample_positions(a.games_dir, a.n, rng)
    print(f"Sampled {len(positions)} positions across regimes", flush=True)

    rows = []
    for pos in positions:
        g = restore(pos['anchor'], 0)
        # legal-move support is model-independent; get indices once from base logits
        obs = torch.from_numpy(_build_obs_for_game(g)).unsqueeze(0).to(dev, dtype)
        with torch.no_grad():
            lb = base(obs)[0].float().cpu().numpy()
        legal = sorted(_get_legal_priors_flat(g.board, lb, 6561).keys())
        if len(legal) < 2:
            continue
        zb = lb[legal] - lb[legal].max(); pb = np.exp(zb); pb /= pb.sum()
        pc, _ = model_dist(corr, dev, dtype, g, legal)
        top1 = int(np.argmax(pb) == np.argmax(pc))
        t3b = set(np.argsort(-pb)[:3]); t3c = set(np.argsort(-pc)[:3])
        rows.append({'ttd': pos['ttd'], 'regime': regime(pos['ttd']),
                     'top1': top1, 'top3': len(t3b & t3c) / 3.0,
                     'js': js_div(pb, pc)})

    # report by regime
    print(f"\n{'regime':>26} {'n':>5} {'top1-agree':>10} {'top3-ovl':>9} "
          f"{'medianJS':>9} {'P90 JS':>8} {'argmax-chg%':>11}", flush=True)
    print('-' * 82, flush=True)
    for _, _, label in BINS:
        rr = [r for r in rows if r['regime'] == label]
        if not rr:
            print(f"{label:>26} {0:>5}"); continue
        t1 = np.mean([r['top1'] for r in rr])
        t3 = np.mean([r['top3'] for r in rr])
        js = np.array([r['js'] for r in rr])
        print(f"{label:>26} {len(rr):>5} {t1:>10.3f} {t3:>9.3f} "
              f"{np.median(js):>9.4f} {np.percentile(js,90):>8.4f} "
              f"{100*(1-t1):>10.1f}%", flush=True)
    allr = rows
    t1all = np.mean([r['top1'] for r in allr])
    print('-' * 82, flush=True)
    print(f"{'ALL':>26} {len(allr):>5} {t1all:>10.3f} "
          f"{np.mean([r['top3'] for r in allr]):>9.3f} "
          f"{np.median([r['js'] for r in allr]):>9.4f}", flush=True)

    json.dump({'meta': vars(a), 'rows': rows}, open(a.out, 'w'))
    print(f"\nWrote {a.out}", flush=True)
    print("\n--- READ ---")
    print("GOOD targeted correction: HEALTHY-bin top1-agree ~1.0 & medianJS ~0 (normal play")
    print("  preserved); drift CONCENTRATED in the crisis bin (the intended fix).")
    print("BAD (forgetting): healthy-bin top1-agree drops / JS rises => correction bled into")
    print("  normal play => smaller task-arith alpha / more rehearsal before shipping.")


if __name__ == '__main__':
    main()
