"""Harvest a DAgger corpus from recorded on-policy games (eval --record-dir).

Three stages in one script:
  1. Parse game JSONs -> candidate states with band metadata. Bands (died games
     only): recovery = last 50 turns, prevention = 51..record_tail turns before
     death; everything else (incl. capped games) = broad. Broad candidates are
     pre-subsampled (--broad-sample) to bound the teacher pass.
  2. Teacher selection pass (fp16 MPS): full-legal argmax + logit gap
     (t_logit[teacher_argmax] - t_logit[recorded_student_move]) per candidate.
  3. Select per the 0b findings (HISTORY 180: value concentrates in confident
     disagreements; gap<0.5 carries ~nothing):
       - band (recovery/prevention) disagreements with gap >= 0.5: all
       - band disagreements gap < 0.5: 25% sample (small dose)
       - broad disagreements gap >= 1.0 (confident only)
       - agreement states: minority sample for soft-label calibration
     Dedup by (board, next_balls); per-game cap. Emits a compact state tensor
     (pol_* zeroed — run distill_relabel to label it) + disagree_mask (int8,
     1 = disagreement with gap >= 0.5) + a meta sidecar (.npz).

    python -m alphatrain.scripts.harvest_dagger_corpus \
        --games-dir data/dagger_games_v1 \
        --teacher alphatrain/data/pillar3k_r3_dw3_T0.7_epoch_22.pt \
        --output alphatrain/data/dagger_v1_states.pt
Then:
    python -m alphatrain.scripts.distill_relabel --teacher <same> \
        --state-tensor alphatrain/data/dagger_v1_states.pt \
        --output alphatrain/data/dagger_v1.pt
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.mcts import _legal_priors_jit
from alphatrain.scripts.distill_relabel import load_teacher

RECOVERY_D = 50


def parse_games(games_dir, broad_sample, rng):
    cand = []  # dicts: board(81 int8), nb, seed, turn, band, student_move
    files = sorted(glob.glob(os.path.join(games_dir, 'game_*.json')))
    n_died = 0
    for gi, fp in enumerate(files):
        d = json.load(open(fp))
        died = d['died']
        n_died += died
        tail = d['record_tail']
        for s in d['states']:
            dist = d['final_turns'] - s['turn'] if died else None
            if died and dist <= RECOVERY_D:
                band = 'recovery'
            elif died and dist <= tail:
                band = 'prevention'
            else:
                band = 'broad'
                if rng.random() > broad_sample:
                    continue
            cand.append({
                'board': np.array(s['board'], dtype=np.int8).reshape(81),
                'nb': [(b['row'], b['col'], b['color'])
                       for b in s['next_balls'][:s['num_next']]],
                'seed': d['seed'], 'turn': s['turn'], 'band': band,
                'student_move': s['move'],
            })
        if (gi + 1) % 500 == 0:
            print(f'  parsed {gi + 1}/{len(files)} games, '
                  f'{len(cand):,} candidates', flush=True)
    print(f'{len(files)} games ({n_died} died), {len(cand):,} candidates',
          flush=True)
    return cand


def to_state_fields(cand):
    n = len(cand)
    boards = torch.zeros((n, 9, 9), dtype=torch.int8)
    next_pos = torch.zeros((n, 3, 2), dtype=torch.int8)
    next_col = torch.zeros((n, 3), dtype=torch.int8)
    n_next = torch.zeros(n, dtype=torch.int8)
    for i, c in enumerate(cand):
        boards[i] = torch.from_numpy(c['board'].reshape(9, 9))
        for t, (r, cc, col) in enumerate(c['nb'][:3]):
            next_pos[i, t, 0] = r
            next_pos[i, t, 1] = cc
            next_col[i, t] = col
        n_next[i] = min(len(c['nb']), 3)
    return boards, next_pos, next_col, n_next


def teacher_pass(cand, fields, teacher_path, device, batch, tmp_path):
    """Full-legal teacher argmax + gap vs the recorded student move."""
    boards, next_pos, next_col, n_next = fields
    nc = boards.shape[0]
    torch.save({'boards': boards, 'next_pos': next_pos, 'next_col': next_col,
                'n_next': n_next,
                'pol_indices': torch.zeros((nc, 5), dtype=torch.int64),
                'pol_values': torch.zeros((nc, 5), dtype=torch.float32),
                'pol_nnz': torch.zeros(nc, dtype=torch.int64),
                'max_score': 0.0}, tmp_path)
    ds = TensorDatasetGPU(tmp_path, augment=False, color_augment=False,
                          augment_factor=1, device=device)
    dev = torch.device(device)
    net = load_teacher(teacher_path, dev, 10, 256)
    dtype = next(net.parameters()).dtype
    n = len(cand)
    t_arg = np.full(n, -1, dtype=np.int64)
    gap = np.zeros(n, dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        obs = ds._build_obs_core(ds.boards[s:e], next_pos=ds.next_pos[s:e],
                                 next_col=ds.next_col[s:e], n_next=ds.n_next[s:e])
        with torch.no_grad():
            lg = net(obs.to(dtype)).float().cpu().numpy()
        bd = ds.boards[s:e].cpu().numpy().astype(np.int8)
        for i in range(e - s):
            k, fi, _ = _legal_priors_jit(bd[i], lg[i], 1)
            if k == 0:
                continue
            t_arg[s + i] = int(fi[0])
            gap[s + i] = lg[i][t_arg[s + i]] - lg[i][cand[s + i]['student_move']]
        if (s // batch) % 20 == 0:
            print(f'  teacher pass {e:,}/{n:,}', flush=True)
    return t_arg, gap


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', default='data/dagger_games_v1')
    p.add_argument('--teacher',
                   default='alphatrain/data/pillar3k_r3_dw3_T0.7_epoch_22.pt')
    p.add_argument('--output', default='alphatrain/data/dagger_v1_states.pt')
    p.add_argument('--device', default='mps')
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--broad-sample', type=float, default=0.25)
    p.add_argument('--low-gap-sample', type=float, default=0.25)
    p.add_argument('--per-game-dis-cap', type=int, default=30)
    p.add_argument('--per-game-low-cap', type=int, default=4)
    p.add_argument('--per-game-agree-cap', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    cand = parse_games(a.games_dir, a.broad_sample, rng)
    fields = to_state_fields(cand)
    t_arg, gap = teacher_pass(cand, fields, a.teacher, a.device, a.batch,
                              a.output + '.candidates.tmp')

    # ---- selection ----
    by_game = {}
    for i, c in enumerate(cand):
        by_game.setdefault(c['seed'], []).append(i)
    band_pri = {'recovery': 0, 'prevention': 1, 'broad': 2}
    selected = []
    for seed, idxs in by_game.items():
        dis_hi, dis_low, agree = [], [], []
        for i in idxs:
            if t_arg[i] < 0:
                continue
            c = cand[i]
            if t_arg[i] == c['student_move']:
                agree.append(i)
            elif c['band'] == 'broad':
                if gap[i] >= 1.0:
                    dis_hi.append(i)
            elif gap[i] >= 0.5:
                dis_hi.append(i)
            elif rng.random() < a.low_gap_sample:
                dis_low.append(i)
        dis_hi.sort(key=lambda i: (band_pri[cand[i]['band']], -gap[i]))
        selected += dis_hi[:a.per_game_dis_cap]
        selected += dis_low[:a.per_game_low_cap]
        rng.shuffle(agree)
        selected += agree[:a.per_game_agree_cap]

    # dedup on (board, next balls)
    seen, final = set(), []
    for i in selected:
        key = (cand[i]['board'].tobytes(), tuple(cand[i]['nb']))
        if key not in seen:
            seen.add(key)
            final.append(i)
    # keep agreements a true minority (<= ~20% of the corpus)
    dis_f = [i for i in final if t_arg[i] != cand[i]['student_move']]
    agr_f = [i for i in final if t_arg[i] == cand[i]['student_move']]
    keep_agr = min(len(agr_f), int(len(dis_f) * 0.25))
    agr_f = list(rng.choice(agr_f, keep_agr, replace=False)) if keep_agr else []
    final = np.array(sorted(dis_f + agr_f))

    n = len(final)
    is_dis = np.array([t_arg[i] != cand[i]['student_move'] for i in final])
    bands = np.array([cand[i]['band'] for i in final])
    print(f'\nselected {n:,} states ({len(selected) - n} dupes dropped):')
    for b in ('recovery', 'prevention', 'broad'):
        m = bands == b
        print(f'  {b:11s}: {m.sum():6,}  '
              f'(disagree {(m & is_dis).sum():6,}, '
              f'gap>=1.0 {(m & is_dis & (gap[final] >= 1.0)).sum():6,})',
              flush=True)
    print(f'  agreements : {(~is_dis).sum():6,} '
          f'({100 * (~is_dis).mean():.0f}%)')

    boards, next_pos, next_col, n_next = fields
    K = 5
    out = {
        'boards': boards[final], 'next_pos': next_pos[final],
        'next_col': next_col[final], 'n_next': n_next[final],
        'pol_indices': torch.zeros((n, K), dtype=torch.int64),
        'pol_values': torch.zeros((n, K), dtype=torch.float32),
        'pol_nnz': torch.zeros(n, dtype=torch.int64),
        'disagree_mask': torch.from_numpy(
            (is_dis & (gap[final] >= 0.5)).astype(np.int8)),
        'num_channels': 18, 'max_score': 0.0, 'value_mode': 'policy_slim',
        'gamma': 0.99, 'num_value_bins': 64,
        'harvest_info': {'games_dir': a.games_dir, 'teacher': a.teacher,
                         'broad_sample': a.broad_sample, 'seed': a.seed},
    }
    torch.save(out, a.output)
    np.savez(a.output.replace('.pt', '_meta.npz'),
             seed=np.array([cand[i]['seed'] for i in final]),
             turn=np.array([cand[i]['turn'] for i in final]),
             band=bands, gap=gap[final], disagree=is_dis,
             student_move=np.array([cand[i]['student_move'] for i in final]),
             teacher_move=t_arg[final])
    os.remove(a.output + '.candidates.tmp')
    print(f'\nwrote {a.output} ({n:,} states, pol_* ZEROED — run '
          f'distill_relabel next) + meta sidecar', flush=True)


if __name__ == '__main__':
    main()
