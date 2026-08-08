"""Generate the H1/H2 discriminator notebooks from the r2bulk_g0 template:
from-scratch (no --resume), 40 epochs, unweighted (GAMMA=0, no weight cell,
no era npz), CHANNELS 128 vs 192. Asserts template structure before editing.

    python -m alphatrain.scripts.gen_scratch_notebooks
"""
import copy
import json

TEMPLATE = 'alphatrain/train_r2bulk_g0_colab.ipynb'
ARMS = {'scratch128': 128, 'scratch192': 192}


def cell_src(c):
    return ''.join(c['source'])


def set_src(c, s):
    c['source'] = s.splitlines(keepends=True)


def main():
    nb0 = json.load(open(TEMPLATE))
    for run, ch in ARMS.items():
        nb = copy.deepcopy(nb0)
        # drop the per-arm weight cell entirely (unweighted control)
        nb['cells'] = [c for c in nb['cells'] if 'ERA_MULT' not in cell_src(c)]
        for c in nb['cells']:
            s = cell_src(c)
            if c['cell_type'] == 'markdown' and s.startswith('# r2bulk_g0'):
                set_src(c, f'# {run} — H1/H2 discriminator: FROM SCRATCH '
                           f'{ch}ch, 40 epochs, UNWEIGHTED r2_bulk (12.51M '
                           'rows). Old-regime recipe retest: val loss is '
                           'meaningful again (unweighted obj = unweighted '
                           'val). Gate stays GAMEPLAY.\n')
            elif '===== CONFIG' in s:
                s = s.replace('# ===== CONFIG (r2bulk_g0) =====',
                              f'# ===== CONFIG ({run}) =====')
                s = s.replace('EPOCHS   = 12', 'EPOCHS   = 40')
                s = s.replace('GAMMA    = 1  # mask carries era*danger; '
                              'trainer w = 1 + mask',
                              'GAMMA    = 0  # unweighted — clean old-regime '
                              'control')
                s = s.replace('RUN      = "r2bulk_g0"', f'RUN      = "{run}"')
                assert 'CHANNELS = 128' in s, 'CHANNELS line moved'
                s = s.replace('CHANNELS = 128', f'CHANNELS = {ch}')
                set_src(c, s)
            elif 'r2_bulk.pt.gz' in s and 'cp' in s:
                # from-scratch: vh3 not needed; drop its copy+assert lines
                s = '\n'.join(l for l in s.splitlines()
                              if 'small128_vh3' not in l) + '\n'
                set_src(c, s)
            elif 'train_path_b' in s:
                s = s.replace('    --resume alphatrain/data/small128_vh3.pt '
                              '--warm-start \\\n', '')
                assert '--resume' not in s, 'resume flag survived'
                set_src(c, s)
            elif 'r2_bulk_era' in s:
                set_src(c, '\n'.join(l for l in s.splitlines()
                                     if 'r2_bulk_era' not in l) + '\n')
        out = f'alphatrain/train_{run}_colab.ipynb'
        cfg = [c for c in nb['cells'] if '===== CONFIG' in cell_src(c)]
        tr = [c for c in nb['cells'] if 'train_path_b' in cell_src(c)]
        assert len(cfg) == 1 and len(tr) == 1, (run, len(cfg), len(tr))
        assert f'CHANNELS = {ch}' in cell_src(cfg[0])
        assert 'EPOCHS   = 40' in cell_src(cfg[0])
        json.dump(nb, open(out, 'w'), indent=1)
        print(f'{out}: {ch}ch, 40ep, from scratch, unweighted')


if __name__ == '__main__':
    main()
