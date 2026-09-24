"""Per-directory score stats for self-play game dirs from FILENAMES (game_seed<S>_score<N>.json),
plus generator metadata from the smallest file in each dir."""
import glob, json, os, re, sys
import numpy as np
dirs = sys.argv[1:]
KEYS = ['policy_model', 'model', 'model_path', 'checkpoint', 'behavior_sims', 'sims', 'num_simulations',
        'max_turns', 'q_weight', 'c_puct', 'value_kind', 'value_module', 'capped', 'turns']
print(f'{"dir":34s} {"n":>5s} {"P50":>7s} {"P25":>7s} {"P75":>7s} {"P90":>8s} {"max":>8s}  meta(smallest file)')
for d in dirs:
    fs = glob.glob(os.path.join(d, 'game_seed*_score*.json'))
    if not fs: continue
    sc = np.array([int(re.search(r'_score(-?\d+)', f).group(1)) for f in fs])
    small = min(fs, key=os.path.getsize)
    meta = {}
    try:
        g = json.load(open(small))
        meta = {k: g[k] for k in KEYS if k in g}
        if 'moves' in g: meta['n_moves'] = len(g['moves'])
        meta['top_keys'] = [k for k in g if k not in ('moves', 'states')][:12]
    except Exception as e:
        meta = {'err': str(e)[:60]}
    print(f'{d:34s} {len(sc):5d} {np.median(sc):7.0f} {np.percentile(sc,25):7.0f} {np.percentile(sc,75):7.0f} '
          f'{np.percentile(sc,90):8.0f} {sc.max():8d}  {meta}')
