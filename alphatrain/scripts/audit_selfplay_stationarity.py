"""Audit turn-band stationarity and diversity in a capped self-play corpus.

This game has no opening/middlegame/endgame progression once strong play
reaches its sustainable regime.  The purpose of this audit is narrower: find
the empirical burn-in after which coarse board/search statistics stabilize,
and verify that a fixed cap is sampling many independent games rather than
being dominated by a few long continuations.

Run large tensors under ``caffeinate -i -s``.  This is CPU/read-only.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def q(values):
    values = np.asarray(values)
    return (f'{values.mean():6.2f} '
            f'[{np.percentile(values, 10):.0f}/'
            f'{np.percentile(values, 50):.0f}/'
            f'{np.percentile(values, 90):.0f}]')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', required=True)
    p.add_argument('--source-name', default='vh3_selfplay_400')
    p.add_argument('--turn-bins', type=int, nargs='+',
                   default=[0, 50, 100, 200, 400, 600, 800, 1000])
    p.add_argument('--unique-sample', type=int, default=50_000)
    p.add_argument('--seed', type=int, default=20260808)
    args = p.parse_args()

    d = torch.load(args.tensor, map_location='cpu', weights_only=True,
                   mmap=True)
    names = (d.get('metadata', {}) or {}).get('source_names', [])
    if args.source_name not in names:
        raise ValueError(f'{args.source_name!r} not in source_names={names}')
    sid = names.index(args.source_name)
    source = d['source_id'].numpy() == sid
    turns = d['turn'].numpy()
    game_id = d['game_id'].numpy()
    selected = np.flatnonzero(source)
    games = np.unique(game_id[selected])
    max_by_game = {}
    for gid, turn in zip(game_id[selected], turns[selected]):
        max_by_game[gid] = max(max_by_game.get(int(gid), -1), int(turn))
    maxima = np.asarray(list(max_by_game.values()))

    print(f'{args.tensor}; source={args.source_name}; '
          f'rows={len(selected):,}; games={len(games):,}', flush=True)
    print(f'max recorded turn/game: {q(maxima)}; '
          f'reached turn 999={100 * (maxima >= 999).mean():.2f}%', flush=True)
    print('turn band   rows   games | empty mean [P10/P50/P90] | colors '
          'same/diff-adj | visit top entropy | behavior=winner | unique',
          flush=True)

    rng = np.random.default_rng(args.seed)
    for lo, hi in zip(args.turn_bins[:-1], args.turn_bins[1:]):
        ix = np.flatnonzero(source & (turns >= lo) & (turns < hi))
        if not len(ix):
            continue
        boards = d['boards'][torch.from_numpy(ix)].numpy()
        empty = (boards == 0).sum((1, 2))
        colors = np.stack(
            [(boards == color).any((1, 2)) for color in range(1, 8)],
            axis=1).sum(1)
        left, right = boards[:, :, :-1], boards[:, :, 1:]
        up, down = boards[:, :-1, :], boards[:, 1:, :]
        h_occ = (left != 0) & (right != 0)
        v_occ = (up != 0) & (down != 0)
        same = ((h_occ & (left == right)).sum((1, 2))
                + (v_occ & (up == down)).sum((1, 2)))
        different = ((h_occ & (left != right)).sum((1, 2))
                     + (v_occ & (up != down)).sum((1, 2)))

        values = d['pol_values'][torch.from_numpy(ix)].float().numpy()
        indices = d['pol_indices'][torch.from_numpy(ix)].numpy()
        top_slot = values.argmax(1)
        top_share = values.max(1)
        entropy = -(np.where(values > 0,
                             values * np.log(np.maximum(values, 1e-30)), 0)
                    .sum(1))
        winner = indices[np.arange(len(ix)), top_slot]
        behavior = d['behavior_move'][torch.from_numpy(ix)].numpy()

        sample = ix
        if len(sample) > args.unique_sample:
            sample = np.sort(rng.choice(
                sample, args.unique_sample, replace=False))
        state_parts = [
            d['boards'][torch.from_numpy(sample)].reshape(len(sample), -1),
            d['next_pos'][torch.from_numpy(sample)].reshape(len(sample), -1),
            d['next_col'][torch.from_numpy(sample)].reshape(len(sample), -1),
            d['n_next'][torch.from_numpy(sample)].reshape(len(sample), 1),
        ]
        packed = torch.cat(state_parts, 1).numpy()
        unique_rate = len(np.unique(packed, axis=0)) / len(sample)

        print(f'{lo:4d}-{hi:<4d} {len(ix):7,d} '
              f'{len(np.unique(game_id[ix])):6,d} | {q(empty)} | '
              f'{colors.mean():5.2f} {same.mean():5.2f}/'
              f'{different.mean():5.2f} | {top_share.mean():5.3f} '
              f'{entropy.mean():5.3f} | '
              f'{100 * (behavior == winner).mean():6.2f}% | '
              f'{100 * unique_rate:6.2f}%', flush=True)


if __name__ == '__main__':
    main()
