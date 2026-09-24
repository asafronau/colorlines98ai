"""Build the ValueHead survive_H tensor (same schema as build_value_targets.py) from games recorded by
`eval --record-dir` (schema: states[{turn,board,next_balls,move}], final_turns, died). Uncapped games ->
no censoring. Keeps every --every-th state plus the full final --tail turns of each game."""
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.scripts.build_value_targets import survive_labels_for_game
from alphatrain.value_head import SURVIVAL_HORIZONS


def main():
    p = argparse.ArgumentParser(); p.add_argument('--games-dir', required=True); p.add_argument('--output', required=True)
    p.add_argument('--every', type=int, default=4); p.add_argument('--tail', type=int, default=300); p.add_argument('--val-frac', type=float, default=0.1)
    a = p.parse_args(); rng = np.random.default_rng(0)
    files = sorted(glob.glob(os.path.join(a.games_dir, 'game_seed*.json')))
    B, NP, NC, NN, L, M, TR = [], [], [], [], [], [], []
    for fi, f in enumerate(files):
        g = json.load(open(f)); st = g['states']; T = g['final_turns']; capped = not g['died']
        labels, masks = survive_labels_for_game(T, capped)
        is_train = rng.random() >= a.val_frac
        for s in st:
            t = s['turn']
            if t < T - a.tail and t % a.every: continue
            B.append(np.array(s['board'], np.int8)); nb = s['next_balls']; k = len(nb)
            npos = np.zeros((3, 2), np.int8); ncol = np.zeros(3, np.int8)
            for i in range(min(k, 3)): npos[i] = (nb[i]['row'], nb[i]['col']); ncol[i] = nb[i]['color']
            NP.append(npos); NC.append(ncol); NN.append(min(k, 3)); L.append(labels[t]); M.append(masks[t]); TR.append(is_train)
        if (fi + 1) % 100 == 0: print(f'  {fi+1}/{len(files)} games, {len(B)} states', flush=True)
    out = {'boards': torch.from_numpy(np.stack(B)), 'next_pos': torch.from_numpy(np.stack(NP)), 'next_col': torch.from_numpy(np.stack(NC)),
           'n_next': torch.from_numpy(np.array(NN, np.int8)), 'survive_labels': torch.from_numpy(np.stack(L).astype(np.int8)),
           'survive_masks': torch.from_numpy(np.stack(M).astype(np.int8)), 'is_train': torch.from_numpy(np.array(TR, bool)), 'horizons': list(SURVIVAL_HORIZONS)}
    torch.save(out, a.output)
    lab = out['survive_labels'].float(); print(f'wrote {a.output}: {len(B):,} states from {len(files)} games; P(survive H) per horizon {lab.mean(0).tolist()}; train frac {np.mean(TR):.2f}')


if __name__ == '__main__':
    main()
