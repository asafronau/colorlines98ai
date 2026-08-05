"""Verify iter5.pt composition (review #7 finding 1) by counting source states.

    python -m alphatrain.scripts.verify_iter5_composition
"""
import glob
import json

def count(d):
    tot = 0
    files = glob.glob(f'{d}/game_seed*.json')
    for fp in files:
        g = json.load(open(fp))
        tot += len(g['moves'])
    return len(files), tot

for d in ('data/crisis_iter5', 'data/selfplay_iter5'):
    nf, ns = count(d)
    print(f'{d}: {nf} games, {ns:,} states')
