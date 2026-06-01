"""Crisis-fork HARVEST: record fresh seeds → mine_crisis_sweep → teacher labels.

Per seed: play to natural death (record), then mine_crisis_sweep (adaptive band
+ batched fp16 + screen/confirm, relative died-within-H metric). Every
(board, next_balls, [(move, catastrophe%)]) row is a teacher label; the
held-out-confirmed forks are the high-value subset. Consolidated to
logs/teacher_labels.json. Time-guarded; harvest from EVERY game's death (the
crisis-onset is in every game, not just early deaths).

    caffeinate -dim python scripts/overnight_systematic.py --n-try 40 \\
        --seed-start 50000 --max-seconds 25000 > logs/harvest.log 2>&1
"""
import os, sys, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MODEL = 'alphatrain/data/pillar3b_epoch_20.pt'
FV_WEIGHTS = 'alphatrain/data/feature_value_weights_2y_nb.npz'

# ── config ──
SEED_START, N_TRY = 50000, 40
REC_CAP, MIN_TURN = 12000, 50          # play to natural death; skip cap-hits
REC_DEVICE = 'mps'                      # fp16 recording (precision irrelevant — re-eval'd fresh)
LO, HI, HORIZON = 15, 45, 300
R_CURVE, R_SCREEN, R_CONFIRM = 100, 100, 500
POL_K, FV_K = 10, 12
MAX_SECONDS = 25000                     # ~7 h
T0 = time.time()


def _argint(flag, default):
    return int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def log(msg):
    print(f"[{(time.time()-T0)/3600:.2f}h] {msg}", flush=True)


def run(cmd, logpath):
    env = dict(os.environ, PYTHONPATH=ROOT)
    with open(logpath, 'a') as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n"); lf.flush()
        return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=ROOT, env=env).returncode


def main():
    global N_TRY, SEED_START, MAX_SECONDS, REC_CAP, LO, HI, R_CURVE, R_SCREEN, R_CONFIRM
    if '--smoke' in sys.argv:
        N_TRY, REC_CAP, SEED_START = 3, 8000, 51000
        LO, HI = 20, 30
        R_CURVE, R_SCREEN, R_CONFIRM = 20, 20, 40
        log("SMOKE MODE: tiny params")
    N_TRY = _argint('--n-try', N_TRY)
    SEED_START = _argint('--seed-start', SEED_START)
    MAX_SECONDS = _argint('--max-seconds', MAX_SECONDS)
    log(f"HARVEST: seeds {SEED_START}..{SEED_START+N_TRY-1}; depths {LO}-{HI}; "
        f"horizon {HORIZON}; R curve/screen/confirm={R_CURVE}/{R_SCREEN}/{R_CONFIRM}; "
        f"rec_cap={REC_CAP}; max={MAX_SECONDS}s")
    os.makedirs('logs', exist_ok=True)
    os.makedirs('alphatrain/data/death_games', exist_ok=True)

    swept = []
    for seed in range(SEED_START, SEED_START + N_TRY):
        if time.time() - T0 > MAX_SECONDS:
            log(f"TIME GUARD ({MAX_SECONDS}s) — stop launching."); break
        gpath = f'alphatrain/data/death_games/death_{seed}.json'
        mpath = f'logs/mine_{seed}.json'
        # play to natural death (fp16) → record trajectory
        run([PY, 'scripts/find_worst_game.py', '--model', MODEL,
             '--record-seed', str(seed), '--max-turns', str(REC_CAP),
             '--device', REC_DEVICE, '--out', gpath], 'logs/harvest_record.log')
        if not os.path.exists(gpath):
            log(f"seed {seed}: record failed; skip."); continue
        g = json.load(open(gpath))
        if not g.get('died') or g['final_turns'] >= REC_CAP or g['final_turns'] < MIN_TURN:
            log(f"seed {seed}: turns={g['final_turns']} died={g.get('died')} — skip.")
            continue
        log(f"seed {seed}: death {g['final_score']}@{g['final_turns']} → mine")
        rc = run([PY, 'scripts/mine_crisis_sweep.py', '--game', gpath,
                  '--model', MODEL, '--fv-weights', FV_WEIGHTS,
                  '--lo', str(LO), '--hi', str(HI), '--horizon', str(HORIZON),
                  '--r-curve', str(R_CURVE), '--r-screen', str(R_SCREEN),
                  '--r-confirm', str(R_CONFIRM), '--pol-k', str(POL_K),
                  '--fv-k', str(FV_K), '--fp16', '--out', mpath],
                 f'logs/harvest_mine_{seed}.log')
        if os.path.exists(mpath):
            swept.append((seed, g, mpath))
        log(f"seed {seed}: mine rc={rc} ({len(swept)} done)")

    # ── consolidate teacher labels + per-seed fork summary ──
    log(f"CONSOLIDATE {len(swept)} swept seeds")
    rows, per_seed, n_real_total = [], [], 0
    for seed, g, mpath in swept:
        try:
            data = json.load(open(mpath))
        except Exception as e:
            log(f"  read fail {mpath}: {e}"); continue
        conf = data.get('confirms', {})
        reals = []
        for r in data['rows']:
            rows.append({'seed': seed, 'depth': r['depth'], 'turn': r['turn'],
                         'empties': r['empties'], 'board': r['board'],
                         'next_balls': r['next_balls'], 'cand_rates': r['cand_rates']})
            c = conf.get(str(r['depth']))
            if r['flag'] and c and c['real']:
                reals.append({'depth': r['depth'], 'gap': round(c['gap'], 1),
                              'pol_cat': c['pol_cat'], 'best_cat': c['best_cat'],
                              'best_move': r['best_move']})
        n_real_total += len(reals)
        per_seed.append({'seed': seed, 'death': [g['final_score'], g['final_turns']],
                         'band': len(data['rows']), 'n_real': len(reals),
                         'real_forks': reals})

    json.dump({'model': MODEL, 'horizon': HORIZON, 'n_rows': len(rows),
               'n_move_labels': sum(len(r['cand_rates']) for r in rows), 'rows': rows},
              open('logs/teacher_labels.json', 'w'), default=float)
    json.dump({'per_seed': per_seed}, open('logs/harvest_summary.json', 'w'),
              indent=1, default=float)

    n = len(swept)
    nfork = sum(1 for s in per_seed if s['n_real'] > 0)
    log("=" * 70)
    log(f"HARVEST DONE: {n} natural-death seeds mined")
    log(f"  teacher rows: {len(rows)}; per-move labels: "
        f"{sum(len(r['cand_rates']) for r in rows)}")
    log(f"  seeds with >=1 REAL fork: {nfork}/{n}; total forks: {n_real_total}")
    for s in per_seed:
        log(f"  seed {s['seed']:>6}: death {s['death'][0]}@{s['death'][1]}; "
            f"band {s['band']}; {s['n_real']} fork(s) "
            f"{[(f['depth'], round(f['gap'])) for f in s['real_forks']]}")
    log("Wrote logs/teacher_labels.json + logs/harvest_summary.json")
    log("DONE.")


if __name__ == '__main__':
    main()
