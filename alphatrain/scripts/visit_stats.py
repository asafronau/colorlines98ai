"""Visit-distribution statistics of full-record self-play games: candidates saved, top-share, visit-argmax vs prior-argmax agreement, by game phase."""
import glob, json, os, sys, numpy as np
d = sys.argv[1]; fs = sorted(glob.glob(os.path.join(d, 'game_seed*.json')))[:int(sys.argv[2]) if len(sys.argv) > 2 else None]
top, agree, ncand, qgap, dead_tail = [], [], [], [], []
for f in fs:
    g = json.load(open(f)); mv = g['moves']; T = len(mv)
    for i, m in enumerate(mv):
        v = np.array(m['cand_visits'], float); p = np.array(m['cand_prior'], float) if 'cand_prior' in m else None
        top.append(v.max() / v.sum()); ncand.append(len(v))
        if p is not None: agree.append(m['cand_moves'][int(v.argmax())] == m['cand_moves'][int(p.argmax())])
        if 'cand_q' in m:
            q = np.array(m['cand_q'], float)[v > 0]; qgap.append(np.sort(q)[-1] - np.sort(q)[-2] if len(q) > 1 else 0)
        dead_tail.append((not g['capped']) and (T - i <= 60))
top, agree, dead_tail = np.array(top), np.array(agree), np.array(dead_tail)
print(f'{len(fs)} games, {len(top):,} moves, cands saved/move {np.mean(ncand):.1f}')
print(f'visit top-share: mean {top.mean():.3f}  P10 {np.percentile(top,10):.2f}  P50 {np.percentile(top,50):.2f}  P90 {np.percentile(top,90):.2f};  share of moves with top-share<0.3: {100*(top<0.3).mean():.1f}%')
if len(agree): print(f'visit argmax == prior argmax: {100*agree.mean():.1f}% overall; last-60-of-dying: {100*agree[dead_tail].mean() if dead_tail.any() else float("nan"):.1f}%')
if qgap: print(f'Q gap top1-top2 among visited: P50 {np.median(qgap):.3f}')
