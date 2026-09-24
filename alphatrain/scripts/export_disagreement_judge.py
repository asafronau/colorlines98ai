"""From the 10x teacher's uncapped games (vh2+MCTS@400+value head), export states where the
student's greedy argmax != the teacher's CHOSEN move, stratified by recorded visit top-share,
to a CLRJ file for rollout_judge (teacher=chosen move, base=student argmax).
Decides: is the student's residual mismatch label noise (near-ties) or unlearned signal?"""
import argparse, glob, json, struct, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model


def flat(m): return (m['sr'] * 9 + m['sc']) * 81 + (m['tr'] * 9 + m['tc'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', default='data/selfplay_iter5')
    p.add_argument('--model', default='alphatrain/data/scratch18b96_lr3e3_ckpts_epoch_40.pt')
    p.add_argument('--per-game', type=int, default=400)
    p.add_argument('--per-bin', type=int, default=700)
    p.add_argument('--skip', type=int, default=15)
    p.add_argument('--out', default='alphatrain/inference_cpp/data/iter5_disagree_states.bin')
    a = p.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for f in sorted(glob.glob(a.games_dir + '/game_seed*.json')):
        g = json.load(open(f)); mv = g['moves']; T = len(mv)
        for t in rng.choice(np.arange(a.skip, T), min(a.per_game, T - a.skip), replace=False):
            m = mv[t]; vis = np.array(m['cand_visits'], float)
            rows.append((np.array(m['board'], np.int8).reshape(81), m['next_balls'], flat(m['chosen_move']),
                         float(vis.max() / vis.sum()), g['seed'], int(t), T - int(t), np.array(m['cand_moves'])))
    print(f'{len(rows)} sampled rows from {len(glob.glob(a.games_dir + "/game_seed*.json"))} games', flush=True)
    dev = torch.device('mps'); net, _ = load_model(a.model, dev, fp16=False)
    obs = np.zeros((len(rows), 18, 9, 9), np.float32)
    for i, r in enumerate(rows):
        nb = r[1]
        obs[i] = build_observation(r[0].reshape(9, 9), np.array([b['row'] for b in nb]), np.array([b['col'] for b in nb]),
                                   np.array([b['color'] for b in nb]), len(nb))
    arg = np.zeros(len(rows), np.int64)
    with torch.inference_mode():
        for s in range(0, len(rows), 2048):
            lg = net(torch.from_numpy(obs[s:s+2048]).to(dev)).float().cpu().numpy()
            for j in range(lg.shape[0]):
                k = rows[s + j][7]; arg[s + j] = k[lg[j][k].argmax()]   # argmax within the teacher's candidate set
    chosen = np.array([r[2] for r in rows]); top = np.array([r[3] for r in rows])
    dis = arg != chosen
    print(f'student argmax != teacher chosen: {dis.sum()} ({100*dis.mean():.2f}%)')
    bins = [(0, .3), (.3, .5), (.5, 1.01)]
    sel, meta = [], []
    for lo, hi in bins:
        m = np.where(dis & (top >= lo) & (top < hi))[0]
        print(f'  top in [{lo},{hi}): {((top>=lo)&(top<hi)).sum()} rows, disagree {len(m)} ({100*len(m)/max(1,((top>=lo)&(top<hi)).sum()):.1f}%)')
        take = rng.choice(m, min(a.per_bin, len(m)), replace=False)
        for i in take: sel.append(i); meta.append({'bin': f'{lo}-{hi}', 'seed': int(rows[i][4]), 'turn': rows[i][5], 'turns_to_end': rows[i][6], 'top': float(top[i])})
    with open(a.out, 'wb') as f:
        f.write(b'CLRJ'); f.write(struct.pack('<i', len(sel)))
        for i in sel:
            b, nb, ch, tp = rows[i][0], rows[i][1], rows[i][2], rows[i][3]
            f.write(b.tobytes()); f.write(struct.pack('<i', len(nb)))
            for t in range(3):
                f.write(struct.pack('<iii', nb[t]['row'], nb[t]['col'], nb[t]['color']) if t < len(nb) else struct.pack('<iii', 0, 0, 0))
            f.write(struct.pack('<iif', int(ch), int(arg[i]), tp))
    json.dump(meta, open(a.out + '.meta.json', 'w'))
    print(f'wrote {a.out}: {len(sel)} disagreement states')


if __name__ == '__main__':
    main()
