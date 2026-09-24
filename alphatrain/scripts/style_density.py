"""Board-occupancy (style) trajectories from eval --record-dir games: mean balls on board at turn
buckets, conditioned on games still alive, plus death-turn percentiles. Compare two models."""
import argparse, glob, json, os
import numpy as np
p = argparse.ArgumentParser(); p.add_argument('dirs', nargs='+'); p.add_argument('--buckets', default='50,100,200,300,500,800,1200')
a = p.parse_args(); B = [int(x) for x in a.buckets.split(',')]
print(f'{"dir":46s} {"games":>5s} {"deathP50":>8s} ' + ' '.join(f'{"balls@"+str(b):>9s}' for b in B) + '   (alive fraction)')
for d in a.dirs:
    fs = sorted(glob.glob(os.path.join(d, 'game_seed*.json'))); occ = {b: [] for b in B}; deaths = []
    for f in fs:
        g = json.load(open(f)); deaths.append(g['final_turns']); st = g['states']
        for b in B:
            if b < len(st): occ[b].append(int((np.array(st[b]['board']) > 0).sum()))
    row = f'{os.path.basename(d):46s} {len(fs):5d} {int(np.median(deaths)):8d} ' + ' '.join(f'{np.mean(occ[b]) if occ[b] else float("nan"):9.1f}' for b in B)
    row += '   ' + ' '.join(f'{len(occ[b])/len(fs):.2f}' for b in B); print(row)
