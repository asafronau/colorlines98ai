"""Stratified sample of relabel flips (teacher visit_argmax != student move) -> CLRJ for rollout_judge."""
import argparse, csv, json, struct
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('--states', required=True); p.add_argument('--csv', required=True); p.add_argument('--out', required=True)
p.add_argument('--per-bin', type=int, default=300); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
rows = list(csv.DictReader(open(a.csv))); meta = json.load(open(a.states + '.meta.json'))
with open(a.states, 'rb') as f:
    assert f.read(4) == b'CLRJ'; n = struct.unpack('<i', f.read(4))[0]; blobs = [f.read(81 + 4 + 36 + 12) for _ in range(n)]
stu = np.array([int(r['file_teacher']) for r in rows]); tea = np.array([int(r['visit_argmax']) for r in rows]); top = np.array([float(r['top_share']) for r in rows])
tte = np.array([m['turns_to_end'] for m in meta]); rng = np.random.default_rng(a.seed)
sel, out_meta = [], []
for lo, hi in ((1, 20), (20, 60), (60, 120), (120, 201)):
    idx = np.where((tea != stu) & (tte >= lo) & (tte < hi))[0]
    for i in rng.choice(idx, min(a.per_bin, len(idx)), replace=False):
        sel.append(int(i)); out_meta.append({**meta[i], 'bin': f'{lo}-{hi}', 'top': float(top[i])})
with open(a.out, 'wb') as f:
    f.write(b'CLRJ'); f.write(struct.pack('<i', len(sel)))
    for i in sel:
        f.write(blobs[i][:121]); f.write(struct.pack('<iif', int(tea[i]), int(stu[i]), float(top[i])))
json.dump(out_meta, open(a.out + '.meta.json', 'w')); print(f'wrote {a.out}: {len(sel)} flips')
