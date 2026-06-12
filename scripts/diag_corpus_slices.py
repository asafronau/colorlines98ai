"""Per-slice provenance diagnostics for the corrections corpus.

mC (13.8k, games 1..1837 by mining chronology) beat mC27k (27.6k, all 3,676 games)
by ~1.8k median on identical recipe+eval, while held-out match stayed ~0.215 for
both — pointing at corpus COMPOSITION, not absorption. This compares the game and
correction distributions across chronological mining slices: source-game final
score and length (were later sessions floor-targeted?), corrections/game, margin
and depth distributions.

    PYTHONPATH=. python scripts/diag_corpus_slices.py
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

CORR_DIR = 'crisis/corrections'
DEATH_DIR = 'crisis/death_games'
BOUNDS = [(0, 1837, 'A 1..1837 (orig mC 13.8k)'),
          (1837, 2593, 'B 1838..2593'),
          (2593, 3042, 'C 2594..3042'),
          (3042, None, 'D 3043.. (newest)')]


def main():
    files = glob.glob(os.path.join(CORR_DIR, 'corr_*.json'))
    files.sort(key=lambda f: os.path.getmtime(f))
    print(f"{len(files)} corr files (mtime-ordered)\n")
    hdr = (f"{'slice':<28} {'games':>5} {'fscore P50':>10} {'fscore P90':>10} "
           f"{'len P50':>8} {'corr/gm':>7} {'dec/gm':>6} {'marg P50':>8} "
           f"{'marg P90':>8} {'depth P50':>9}")
    print(hdr)
    print('-' * len(hdr))
    for lo, hi, name in BOUNDS:
        chunk = files[lo:hi]
        if not chunk:
            continue
        fscores, lengths, n_corr, n_dec, margins, depths = [], [], 0, 0, [], []
        for f in chunk:
            try:
                d = json.load(open(f))
            except Exception:
                continue
            seed = d['seed']
            dg = os.path.join(DEATH_DIR, f'death_{seed}.json')
            if os.path.exists(dg):
                g = json.load(open(dg))
                fscores.append(g.get('final_score', -1))
                if g.get('frames'):
                    lengths.append(g['frames'][-1].get('turn', -1))
            for c in d['corrections']:
                n_corr += 1
                m = c['mcts_top_share'] - c['pol_share']
                if m >= 0.05:
                    n_dec += 1
                    margins.append(m)
                    depths.append(c['depth'])
        fs, ln = np.array(fscores), np.array(lengths)
        mg, dp = np.array(margins), np.array(depths)
        print(f"{name:<28} {len(chunk):>5} "
              f"{np.percentile(fs, 50) if len(fs) else -1:>10.0f} "
              f"{np.percentile(fs, 90) if len(fs) else -1:>10.0f} "
              f"{np.percentile(ln, 50) if len(ln) else -1:>8.0f} "
              f"{n_corr/len(chunk):>7.1f} {n_dec/len(chunk):>6.1f} "
              f"{np.percentile(mg, 50) if len(mg) else -1:>8.3f} "
              f"{np.percentile(mg, 90) if len(mg) else -1:>8.3f} "
              f"{np.percentile(dp, 50) if len(dp) else -1:>9.0f}")
    # Death-game availability check (mining may span dirs)
    missing = sum(1 for f in files
                  if not os.path.exists(os.path.join(
                      DEATH_DIR, f"death_{json.load(open(f))['seed']}.json")))
    if missing:
        print(f"\nNOTE: {missing} corr files have no matching death_*.json "
              f"in {DEATH_DIR} (their fscore/len rows exclude them)")


if __name__ == '__main__':
    main()
