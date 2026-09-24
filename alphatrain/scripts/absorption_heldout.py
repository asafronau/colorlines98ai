"""Did distillation actually install the search's choices? On HELD-OUT full-record search games, compare each
model's greedy (legal) argmax with the search's chosen move, split by whether the search overrode the recorded
prior (the base actor's argmax). Also by board occupancy."""
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit


def flat(m): return (m['sr'] * 9 + m['sc']) * 81 + m['tr'] * 9 + m['tc']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--games-dir', required=True); p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--every', type=int, default=3); a = p.parse_args()
    boards, obs, chosen, prior_arg, occ = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(a.games_dir, 'game_seed*.json'))):
        g = json.load(open(f))
        for i, m in enumerate(g['moves']):
            if i % a.every: continue
            b = np.array(m['board'], np.int8); nb = m['next_balls'][:m['num_next']]
            boards.append(b); chosen.append(flat(m['chosen_move']))
            prior_arg.append(m['cand_moves'][int(np.argmax(m['cand_prior']))]); occ.append(int((b > 0).sum()))
            obs.append(build_observation(b, np.array([x['row'] for x in nb]), np.array([x['col'] for x in nb]), np.array([x['color'] for x in nb]), len(nb)))
    chosen, prior_arg, occ = np.array(chosen), np.array(prior_arg), np.array(occ); obs = torch.from_numpy(np.stack(obs).astype(np.float32))
    over = chosen != prior_arg
    print(f'{len(chosen):,} held-out states; search overrode base prior on {100*over.mean():.1f}%')
    bins = [(0, 40), (40, 50), (50, 60), (60, 82)]
    print(f'{"model":40s} {"agree(all)":>10s} {"on agree-rows":>13s} {"ADOPT overrides":>15s}  ' + ' '.join(f'adopt occ{lo}-{hi}'.rjust(15) for lo, hi in bins))
    dev = torch.device('mps')
    for mp in a.models:
        net, _ = load_model(mp, dev, fp16=False); arg = np.zeros(len(chosen), np.int64)
        with torch.inference_mode():
            for s in range(0, len(chosen), 2048):
                lg = net(obs[s:s+2048].to(dev)).float().cpu().numpy()
                for j in range(lg.shape[0]):
                    cnt, idx, _ = _legal_priors_jit(boards[s + j], lg[j].astype(np.float32), 1); arg[s + j] = idx[0] if cnt else -1
        hit = arg == chosen
        row = f'{os.path.basename(mp)[:40]:40s} {100*hit.mean():10.2f} {100*hit[~over].mean():13.2f} {100*hit[over].mean():15.2f}  '
        row += ' '.join(f'{100*hit[over & (occ>=lo) & (occ<hi)].mean():14.1f}%' if (over & (occ>=lo) & (occ<hi)).sum() > 30 else '             - ' for lo, hi in bins)
        print(row, flush=True)
    print('occupancy of override rows:', {f'{lo}-{hi}': int((over & (occ>=lo) & (occ<hi)).sum()) for lo, hi in bins})


if __name__ == '__main__':
    main()
