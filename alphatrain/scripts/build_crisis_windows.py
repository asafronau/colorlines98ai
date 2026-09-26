"""Cut mcts_crisis replays down to their crisis windows (the actual corrections).

Each replay rewinds `rewind_turns` before the actor's death, plays with search, and (if it escapes)
continues up to `continue_turns`. Most rows are that quiet continuation; this keeps only the moves
up to the original death turn plus the next --after moves, optionally only from escaped replays
(outcome-verified: the search line survived the whole continuation). Writes truncated copies of
the game JSONs, ready for build_expert_v2_tensor --policy-only-data.

    python -m alphatrain.scripts.build_crisis_windows --in-dir data/gen1_crisis_A1 \
        --out-dir data/gen1_crisis_A1_windows --after 15 --escaped-only
"""
import argparse
import collections
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--after', type=int, default=15, help='moves kept past the original death turn')
    ap.add_argument('--escaped-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    kept, rows = collections.Counter(), collections.Counter()
    for f in sorted(glob.glob(os.path.join(a.in_dir, 'game_seed*.json'))):
        g = json.load(open(f))
        escaped = g['turns'] - g['replay_from_turn'] >= g['continue_turns']
        assert escaped == g['capped'], f'{f}: escape status disagrees with the capped flag'
        if a.escaped_only and not escaped:
            continue
        g['moves'] = g['moves'][:g['rewind_turns'] + a.after]
        g['window_after'] = a.after
        kept[g['label']] += 1
        rows[g['label']] += len(g['moves'])
        with open(os.path.join(a.out_dir, os.path.basename(f)), 'w') as out:
            json.dump(g, out)
    print(f'kept replays {dict(kept)}; rows {dict(rows)}; total rows {sum(rows.values()):,}', flush=True)


if __name__ == '__main__':
    main()
