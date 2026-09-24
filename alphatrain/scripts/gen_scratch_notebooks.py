"""Generate controlled from-scratch recipe notebooks from r2bulk_g0.

The original H1/H2 discriminator compares 128ch with 192ch at one optimizer
recipe.  The additional 128ch arms separate update-count starvation from
per-step learning-rate starvation and include a historical-geometry bridge.
All notebooks fail fast if the Colab tarball predates the exact D4 feature
transform: six of eight old transforms rotated directional feature maps without
permuting their H/V/D1/D2 channel meanings.

    python -m alphatrain.scripts.gen_scratch_notebooks
"""
import argparse
import copy
import json

TEMPLATE = 'alphatrain/train_r2bulk_g0_colab.ipynb'
TARBALL = 'colorlines_pillar3d_v5.tar.gz'
ARMS = {
    # Upper point on the corrected high-LR width curve.  This is also the
    # exact 10b x 256ch architecture used by pillar3k, but trained from scratch
    # on the controlled R2 corpus/recipe so data lineage and width separate.
    'scratch256_lr3e3': {'blocks': 10, 'channels': 256, 'epochs': 40,
                         'batch': 32768, 'lr': '3e-3',
                         'save_steps': 1000,
                         'purpose': 'upper-width scaling control'},
    # Parameter-matched depth/width control for 10b x 128ch (3.003M params).
    # At 3.032M params this asks whether extra sequential computation can
    # substitute for width, especially on crisis/clearance patterns.
    'scratch18b96_lr3e3': {'blocks': 18, 'channels': 96, 'epochs': 40,
                           'batch': 32768, 'lr': '3e-3',
                           'save_steps': 1000,
                           'purpose': 'parameter-matched depth control'},
    # Third point on the corrected high-LR width curve.  Keep the backbone
    # depth and policy head unchanged so this isolates hidden width.
    'scratch96_lr3e3': {'blocks': 10, 'channels': 96, 'epochs': 40,
                        'batch': 32768, 'lr': '3e-3', 'save_steps': 1000,
                        'purpose': 'width-scaling control'},
    # Existing width discriminator.
    'scratch128': {'channels': 128, 'epochs': 40,
                   'batch': 32768, 'lr': '3e-4', 'save_steps': 250},
    'scratch192': {'channels': 192, 'epochs': 40,
                   'batch': 32768, 'lr': '3e-4', 'save_steps': 250},
    # Clean baseline for the post-D4-fix recipe arms.  Keep a distinct run
    # name: the existing scratch128 checkpoints were trained with the broken
    # directional-channel D4 transform and must never share a Drive folder.
    'scratch128_d4ctrl': {'channels': 128, 'epochs': 40,
                          'batch': 32768, 'lr': '3e-4', 'save_steps': 1000},
    # Same LR, 16x more optimizer updates per corpus pass.
    # Sixteen epochs already slightly exceed the historical launch pad's
    # complete ~714k-update budget; extend only after reading that curve.
    'scratch128_bs2048': {'channels': 128, 'epochs': 16,
                          'batch': 2048, 'lr': '3e-4', 'save_steps': 5000},
    # Same update count, 10x larger peak step.
    'scratch128_lr3e3': {'channels': 128, 'epochs': 40,
                         'batch': 32768, 'lr': '3e-3', 'save_steps': 1000},
    # Closest transfer of the successful 128ch launch-pad optimizer geometry.
    # Roughly 31 current-corpus epochs match its ~714k updates and ~2.9B rows.
    'scratch128_histgeom': {'channels': 128, 'epochs': 31,
                            'batch': 4096, 'lr': '1e-3', 'save_steps': 5000},
}


def cell_src(c):
    return ''.join(c['source'])


def set_src(c, s):
    c['source'] = s.splitlines(keepends=True)


def gate_markdown(run):
    """Distribution-only evaluation instructions for scratch width arms."""
    return f"""## Evaluation protocol (local M5, C++ FP16)

Download approximately every third epoch first (3/6/9/12/18/24/30/36/40).
Convert each checkpoint directly to its own TorchScript path:

```bash
source .venv/bin/activate
python -m alphatrain.inference_cpp.export_ts \\
  --model alphatrain/data/{run}_ckpts_epoch_12.pt \\
  --output alphatrain/inference_cpp/data/{run}_ckpts_epoch_12_ts.pt
```

Run aggregate-only 999-game screens; per-game traces remain off by default:

```bash
cd alphatrain/inference_cpp
caffeinate -i -s ./build/eval \\
  --model data/{run}_ckpts_epoch_12_ts.pt --device mps --batch 500 \\
  --seed-start 775000 --seed-end 775999 \\
  --scores-out data/{run}_ep12_775k_999.csv
```

Take only the best apparent plateau checkpoint(s) to 5,000 games by changing
`--seed-end` to `780000`. Compare aggregate distributions independently: P5/P10,
`<1000`, median, upper tail, and survival horizons. Matching seed identifiers are
environmental controls, never a paired-game statistic. Reserved final seeds
1,100,000--1,119,999 remain untouched.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'arms', nargs='*', metavar='ARM',
        help='Generate only these arms (default: generate every arm).')
    args = parser.parse_args()
    unknown = sorted(set(args.arms) - set(ARMS))
    if unknown:
        parser.error(f"unknown arm(s): {', '.join(unknown)}; "
                     f"choose from {', '.join(sorted(ARMS))}")
    selected = args.arms or list(ARMS)

    nb0 = json.load(open(TEMPLATE))
    for run in selected:
        cfg = ARMS[run]
        blocks = cfg.get('blocks', 10)
        ch = cfg['channels']
        nb = copy.deepcopy(nb0)
        # drop the per-arm weight cell entirely (unweighted control)
        nb['cells'] = [c for c in nb['cells'] if 'ERA_MULT' not in cell_src(c)]
        for c in nb['cells']:
            s = cell_src(c)
            if c['cell_type'] == 'markdown' and s.startswith('# r2bulk_g0'):
                purpose = cfg.get('purpose', 'H1/H2 discriminator')
                set_src(c, f'# {run} — {purpose}: FROM SCRATCH '
                           f'{blocks}b x '
                           f"{ch}ch, {cfg['epochs']} epochs, UNWEIGHTED "
                           'r2_bulk (12.51M '
                           'rows). Objective and validation are internally '
                           'matched controls, but raw val loss is informational '
                           'only. Gate stays GAMEPLAY.\n')
            elif '===== CONFIG' in s:
                s = s.replace('# ===== CONFIG (r2bulk_g0) =====',
                              f'# ===== CONFIG ({run}) =====')
                s = s.replace('EPOCHS   = 12',
                              f"EPOCHS   = {cfg['epochs']}")
                s = s.replace('BATCH    = 32768',
                              f"BATCH    = {cfg['batch']}")
                s = s.replace('LR       = 3e-4',
                              f"LR       = {cfg['lr']}")
                s = s.replace('SAVE_STEPS = 250',
                              f"SAVE_STEPS = {cfg['save_steps']}")
                s = s.replace('GAMMA    = 1  # mask carries era*danger; '
                              'trainer w = 1 + mask',
                              'GAMMA    = 0  # unweighted — clean old-regime '
                              'control')
                s = s.replace('RUN      = "r2bulk_g0"', f'RUN      = "{run}"')
                assert 'CHANNELS = 128' in s, 'CHANNELS line moved'
                s = s.replace('CHANNELS = 128',
                              f'NUM_BLOCKS = {blocks}\nCHANNELS = {ch}')
                set_src(c, s)
            elif 'r2_bulk.pt.gz' in s and 'cp' in s:
                # from-scratch: vh3 not needed; drop its copy+assert lines
                s = '\n'.join(l for l in s.splitlines()
                              if 'small128_vh3' not in l) + '\n'
                s = s.replace('colorlines_pillar3d_v5.tar.gz', TARBALL)
                guard = (
                    "\n# Refuse the historical corrupt D4 implementation.\n"
                    "_dataset_source = open('/content/alphatrain/dataset.py').read()\n"
                    "assert ('def _build_line_direction_luts' in _dataset_source and\n"
                    "        'def _transform_observation' in _dataset_source), (\n"
                    "    'STALE CODE TARBALL: directional H/V/D1/D2 channels are not '\n"
                    "    'transformed under D4; rebuild/upload " + TARBALL + ".')\n"
                    "print('Exact directional-channel D4 transform confirmed.')\n")
                s += guard
                set_src(c, s)
            elif 'train_path_b' in s:
                s = s.replace('    --resume alphatrain/data/small128_vh3.pt '
                              '--warm-start \\\n', '')
                assert '--resume' not in s, 'resume flag survived'
                assert '--channels {CHANNELS}' in s, 'channels flag moved'
                s = s.replace('--channels {CHANNELS}',
                              '--num-blocks {NUM_BLOCKS} '
                              '--channels {CHANNELS}')
                resume = (
                    "# Resume from the last COMPLETED epoch if Colab stopped.\n"
                    "LATEST = (f'/content/drive/MyDrive/alphatrain/"
                    "{RUN}_ckpts/latest.pt')\n"
                    "RESUME_ARGS = f'--resume {LATEST}' if "
                    "os.path.exists(LATEST) else ''\n"
                    "print('Resume:', LATEST if RESUME_ARGS else "
                    "'none (new run)')\n")
                s = resume + s
                s = s.replace('    --tensor-file',
                              '    {RESUME_ARGS} --tensor-file')
                set_src(c, s)
            elif 'r2_bulk_era' in s:
                set_src(c, '\n'.join(l for l in s.splitlines()
                                     if 'r2_bulk_era' not in l) + '\n')
            elif (c['cell_type'] == 'markdown'
                  and s.startswith('## Gate protocol')):
                set_src(c, gate_markdown(run))
        out = f'alphatrain/train_{run}_colab.ipynb'
        cfg_cells = [c for c in nb['cells']
                     if '===== CONFIG' in cell_src(c)]
        tr = [c for c in nb['cells'] if 'train_path_b' in cell_src(c)]
        assert len(cfg_cells) == 1 and len(tr) == 1, (
            run, len(cfg_cells), len(tr))
        assert f'NUM_BLOCKS = {blocks}' in cell_src(cfg_cells[0])
        assert f'CHANNELS = {ch}' in cell_src(cfg_cells[0])
        assert f"EPOCHS   = {cfg['epochs']}" in cell_src(cfg_cells[0])
        assert f"BATCH    = {cfg['batch']}" in cell_src(cfg_cells[0])
        assert f"LR       = {cfg['lr']}" in cell_src(cfg_cells[0])
        assert f"SAVE_STEPS = {cfg['save_steps']}" in cell_src(cfg_cells[0])
        json.dump(nb, open(out, 'w'), indent=1)
        assert '--num-blocks {NUM_BLOCKS}' in cell_src(tr[0])
        assert '{RESUME_ARGS} --tensor-file' in cell_src(tr[0])
        print(f"{out}: {blocks}b x {ch}ch, {cfg['epochs']}ep, "
              f"bs={cfg['batch']}, "
              f"lr={cfg['lr']}, from scratch, unweighted")


if __name__ == '__main__':
    main()
