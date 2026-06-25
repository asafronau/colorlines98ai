"""Quiet-gap pilot: is a stable-play inaccuracy detectable above RNG noise?

The corrected instrument (vs rewind_from_death, which is a crisis/died-within-H detector):
  - METRIC = continuous turns-survived to REAL death (no hardcoded board metric; survival
    is the only judge). Rolling each candidate to death and comparing is one step of policy
    iteration — a judge-free signal stronger than the policy's one-step prior.
  - RNG-ROBUST = each candidate scored over many seeds (different spawn futures), then the
    best alternative is RE-CHECKED on FRESH held-out seeds with a paired bootstrap CI that
    must exclude 0 (the 4800-sim winner's-curse lesson). Common-RNG = candidates share the
    seed SET (paired variance reduction), NOT a single seed.
  - "QUIET" is MODEL-DEFINED: report pol_cat = policy's own died-within-horizon rate at the
    anchor. Genuinely quiet = pol_cat ~ 0. (Board occupancy is reported for context only,
    never used as a metric or target.)

Two populations (both sampled from the death-game corpus, which carries full frames):
  - HEALTHY: an EARLY frame of a LONG game (far from death) = representative stable play.
            Expect rollouts to censor at the horizon (death unreachable) -> tells us whether
            a single quiet move's survival effect is detectable at all at feasible horizons.
  - DEATH-QUIET: a frame ~PRE turns BEFORE the eventual death = a stable-looking decision on
            a trajectory that did die. Death is reachable -> a REAL fork here = the earlier
            quiet move WAS avoidable (your accumulation hypothesis, confirmed & mineable).

    PYTHONPATH=. python scripts/quiet_gap_pilot.py --n-healthy 6 --n-death 6 \
        --top-k 3 --disc 24 --held 24 --horizon 1200 --fp32 --out /tmp/quiet_pilot.json
"""
import os, sys, glob, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.evaluate import load_model
from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
from scripts.batched_rollout import restore, batched_rollout, _decode


def topk_candidates(net, dev, dtype, anchor, k):
    """Policy's top-k legal moves (decoded). cand[0] = argmax."""
    g = restore(anchor, 0)
    obs = torch.from_numpy(_build_obs_for_game(g)).unsqueeze(0).to(dev, dtype)
    with torch.no_grad():
        logits = net(obs)[0].float().cpu().numpy()
    pri = _get_legal_priors_flat(g.board, logits, 30)
    if len(pri) < 2:
        return None
    ranked = sorted(pri.items(), key=lambda x: -x[1])[:k]
    return [_decode(m) for m, _ in ranked]


def frame_anchor(fr):
    return {'board': fr['board'], 'next_balls': fr['next_balls'],
            'score': fr.get('score_before', fr['score']), 'turn': fr['turn']}


def select_anchors(games_dir, n_healthy, n_death, *, long_min, healthy_turn,
                   death_lo, death_hi, pre):
    """Pick HEALTHY (early frame of long games) and DEATH-QUIET (pre-death frame of
    medium games) anchors. Returns list of (label, seed, frame_idx, anchor)."""
    files = sorted(glob.glob(os.path.join(games_dir, 'death_*.json')))
    healthy, death = [], []
    for f in files:
        if len(healthy) >= n_healthy and len(death) >= n_death:
            break
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ft, frames, seed = d.get('final_turns', 0), d.get('frames', []), d.get('seed')
        if not frames:
            continue
        # HEALTHY: early frame of a long game (far from the eventual death)
        if len(healthy) < n_healthy and ft >= long_min:
            ti = min(range(len(frames)), key=lambda i: abs(frames[i]['turn'] - healthy_turn))
            fr = frames[ti]
            if fr['turn'] >= 40 and (ft - fr['turn']) >= 1500:
                healthy.append(('healthy', seed, ti, frame_anchor(fr)))
                continue
        # DEATH-QUIET: frame ~pre turns before death, on a medium-length doomed game
        if len(death) < n_death and death_lo <= ft <= death_hi:
            target = ft - pre
            ti = min(range(len(frames)), key=lambda i: abs(frames[i]['turn'] - target))
            fr = frames[ti]
            if fr['turn'] >= 40 and (ft - fr['turn']) >= 60:
                death.append(('death_quiet', seed, ti, frame_anchor(fr)))
    return healthy + death


def boot_ci(delta, rng, n_boot=2000):
    n = len(delta)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        bs[b] = delta[rng.integers(0, n, n)].mean()
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--games-dir', default='alphatrain/data/death_games')
    p.add_argument('--n-healthy', type=int, default=6)
    p.add_argument('--n-death', type=int, default=6)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--disc', type=int, default=24, help='discovery seeds')
    p.add_argument('--held', type=int, default=24, help='held-out fresh seeds')
    p.add_argument('--horizon', type=int, default=1200)
    p.add_argument('--metric', choices=['turns', 'catastrophe'], default='catastrophe',
                   help="'catastrophe' (RIGHT objective, infinite-game theory) = flag/held-out "
                        "on P(died-within-horizon) reduction (left-tail risk). 'turns' = mean "
                        "survival delta (washes out the tail; underpowered — see memory).")
    p.add_argument('--flag-turns', type=float, default=20.0,
                   help='turns metric: flag if best alt out-survives argmax by >= this many turns')
    p.add_argument('--flag-pp', type=float, default=5.0,
                   help='catastrophe metric: flag if best alt LOWERS catastrophe rate by >= this pp')
    p.add_argument('--long-min', type=int, default=3000)
    p.add_argument('--healthy-turn', type=int, default=600)
    p.add_argument('--death-lo', type=int, default=250)
    p.add_argument('--death-hi', type=int, default=1200)
    p.add_argument('--pre', type=int, default=150, help='death-quiet: turns before death')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--device', default=None)
    p.add_argument('--out', default='/tmp/quiet_pilot.json')
    a = p.parse_args()

    dev = torch.device(a.device) if a.device else torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available() else 'cpu')
    net, _ = load_model(a.model, dev, fp16=not a.fp32)
    dtype = next(net.parameters()).dtype
    print(f"Device {dev} dtype {dtype} | model {a.model}", flush=True)

    anchors = select_anchors(a.games_dir, a.n_healthy, a.n_death,
                             long_min=a.long_min, healthy_turn=a.healthy_turn,
                             death_lo=a.death_lo, death_hi=a.death_hi, pre=a.pre)
    nh = sum(1 for x in anchors if x[0] == 'healthy')
    nd = sum(1 for x in anchors if x[0] == 'death_quiet')
    print(f"Selected {nh} healthy + {nd} death-quiet anchors", flush=True)

    # candidates per anchor
    recs = []
    for label, seed, ti, anc in anchors:
        cand = topk_candidates(net, dev, dtype, anc, a.top_k)
        if cand is None:
            continue
        recs.append({'label': label, 'seed': seed, 'turn': anc['turn'],
                     'empties': int(np.sum(np.array(anc['board']) == 0)),
                     'anchor': anc, 'cand': cand})
    print(f"{len(recs)} anchors with >=2 legal candidates", flush=True)

    # ── DISCOVERY: all anchors x candidates x disc seeds, chunked for progress ──
    disc_jobs, job_map = [], []
    for ri, r in enumerate(recs):
        for ci, c in enumerate(r['cand']):
            for s in range(a.disc):
                disc_jobs.append((r['anchor'], c, s)); job_map.append((ri, ci, s))
    print(f"Discovery: {len(disc_jobs)} rollouts (horizon {a.horizon})", flush=True)
    t0 = time.time()
    res = [None] * len(disc_jobs)
    CHUNK = max(a.batch * 8, 256)
    for st in range(0, len(disc_jobs), CHUNK):
        out = batched_rollout(net, dev, dtype, disc_jobs[st:st+CHUNK], a.horizon, batch=a.batch)
        res[st:st+len(out)] = out
        el = time.time() - t0
        print(f"  [disc {min(st+CHUNK,len(disc_jobs))}/{len(disc_jobs)}] {el:.0f}s", flush=True)

    # aggregate discovery per (anchor, candidate)
    turns = np.zeros((len(recs), a.top_k, a.disc))
    died = np.zeros((len(recs), a.top_k, a.disc))
    for (ri, ci, s), rr in zip(job_map, res):
        turns[ri, ci, s] = rr['turns']; died[ri, ci, s] = rr['died']
    flagged = []
    for ri, r in enumerate(recs):
        nc = len(r['cand'])
        cat = 100 * died[ri, :nc].mean(1)                 # catastrophe rate per candidate (pp)
        mt = turns[ri, :nc].mean(1)                       # mean turns per candidate
        r['pol_cat'] = float(cat[0])                      # argmax died-within-H rate (danger)
        r['censor'] = float(100 * (turns[ri, 0] >= a.horizon).mean())
        r['argmax_turns'] = float(mt[0])
        if nc > 1:
            if a.metric == 'catastrophe':
                alt = 1 + int(np.argmin(cat[1:]))         # safest alternative (lowest cat rate)
                r['best_alt'] = alt
                r['disc_gap'] = float(cat[0] - cat[alt])  # pp catastrophe REDUCED by alt (>0 safer)
                if r['disc_gap'] >= a.flag_pp:
                    flagged.append(ri)
            else:
                alt = 1 + int(np.argmax(mt[1:]))          # longest-surviving alternative
                r['best_alt'] = alt
                r['disc_gap'] = float(mt[alt] - mt[0])
                if r['disc_gap'] >= a.flag_turns:
                    flagged.append(ri)
        else:
            r['best_alt'] = None; r['disc_gap'] = 0.0
    thr = f">= {a.flag_pp}pp lower catastrophe" if a.metric == 'catastrophe' else f">= {a.flag_turns}t longer survival"
    print(f"\nDiscovery done ({time.time()-t0:.0f}s). {len(flagged)} anchors flagged "
          f"(best alt {thr} in discovery)", flush=True)

    # ── HELD-OUT: re-check flagged on FRESH seeds, paired bootstrap CI ──
    held_jobs, held_map = [], []
    for ri in flagged:
        for ci in (0, recs[ri]['best_alt']):
            for s in range(a.disc, a.disc + a.held):
                held_jobs.append((recs[ri]['anchor'], recs[ri]['cand'][ci], s))
                held_map.append((ri, ci, s))
    rng = np.random.default_rng(0)
    if held_jobs:
        print(f"Held-out: {len(held_jobs)} rollouts on fresh seeds", flush=True)
        hres = [None] * len(held_jobs)
        for st in range(0, len(held_jobs), CHUNK):
            out = batched_rollout(net, dev, dtype, held_jobs[st:st+CHUNK], a.horizon, batch=a.batch)
            hres[st:st+len(out)] = out
        ht = {}
        for (ri, ci, s), rr in zip(held_map, hres):
            ht.setdefault((ri, ci), {})[s] = (rr['turns'], rr['died'])
        for ri in flagged:
            alt = recs[ri]['best_alt']
            pol_d, alt_d = ht.get((ri, 0), {}), ht.get((ri, alt), {})
            seeds = sorted(set(pol_d) & set(alt_d))
            if not seeds:
                continue
            if a.metric == 'catastrophe':
                # paired died-flag delta: argmax_died - alt_died per seed (common RNG).
                # >0 = argmax died where alt survived = alt is SAFER. *100 -> pp.
                delta = np.array([100.0 * (pol_d[s][1] - alt_d[s][1]) for s in seeds], float)
            else:
                delta = np.array([alt_d[s][0] - pol_d[s][0] for s in seeds], float)  # alt-argmax turns
            lo, hi = boot_ci(delta, rng)
            recs[ri]['held'] = {'gap': float(delta.mean()), 'lo': lo, 'hi': hi,
                                'n': len(seeds), 'survives': bool(lo > 0)}

    # ── REPORT ──
    is_cat = a.metric == 'catastrophe'
    unit = 'pp' if is_cat else 't'
    flag_thr = a.flag_pp if is_cat else a.flag_turns
    gaplabel = 'discΔpp' if is_cat else 'discΔt'
    print(f"\n{'pop':>12} {'turn':>5} {'empt':>4} {'polCat%':>7} {'censor%':>7} "
          f"{gaplabel:>8}  {'verdict':>26}", flush=True)
    print('-' * 87, flush=True)
    n_real = {'healthy': 0, 'death_quiet': 0}
    for r in recs:
        v = 'neutral'
        if r['disc_gap'] >= flag_thr:
            h = r.get('held')
            if h is None:
                v = 'flag(no-held)'
            elif h['survives']:
                v = f"REAL Δ{h['gap']:.0f}{unit}[{h['lo']:.0f},{h['hi']:.0f}]"; n_real[r['label']] += 1
            else:
                v = f"curse Δ{h['gap']:.0f}{unit}[{h['lo']:.0f},{h['hi']:.0f}]"
        print(f"{r['label']:>12} {r['turn']:>5} {r['empties']:>4} {r['pol_cat']:>7.1f} "
              f"{r['censor']:>7.0f} {r['disc_gap']:>8.0f}  {v:>26}", flush=True)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({'meta': vars(a), 'recs': [{k: v for k, v in r.items() if k != 'anchor'}
                                         for r in recs]}, open(a.out, 'w'), default=str)
    print(f"\nREAL stable-play forks: healthy {n_real['healthy']}/{nh}, "
          f"death_quiet {n_real['death_quiet']}/{nd}. Wrote {a.out}", flush=True)
    print("\n--- READ ---")
    print("REAL fork (held-out CI excludes 0) on a GENUINELY QUIET anchor (polCat~0) =>")
    print("  a stable-play inaccuracy is detectable above RNG noise => mineable (targeted).")
    print("All 'curse'/'neutral' on quiet anchors, esp. with high censor% => single quiet")
    print("  moves have no rollout-detectable survival delta at this horizon => accumulation")
    print("  is NOT attackable move-by-move; need whole-game or deeper-horizon approach.")


if __name__ == '__main__':
    main()
