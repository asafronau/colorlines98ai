"""Task-arithmetic stage 2: θ_deploy = θ_base + α · (θ_crisis − θ_base).

Interpolates float parameters between the base policy and the corrections-fine-tuned
policy (scripts/train_crisis_ft.py). BN running stats / num_batches_tracked are taken
verbatim from BASE and verified identical in the fine-tune (frozen_bn) — one
normalization regime throughout. Output loads with the standard eval scripts.

Sweep α with the cheap local eval — each merge costs seconds:
    for A in 0.05 0.1 0.2 0.4; do
        PYTHONPATH=. python scripts/merge_checkpoints.py \\
            --base alphatrain/data/pillar3b_epoch_20.pt \\
            --crisis checkpoints/crisis_ft/ft_epoch_20.pt --alpha $A \\
            --out alphatrain/data/ta_a$A.pt
        PYTHONPATH=. python -m scripts.eval_policy --model alphatrain/data/ta_a$A.pt \\
            --seed-start 775000 --seed-end 775999 --device mps --batch 256
    done
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def state_of(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    st = ck['model']
    if any(k.startswith('_orig_mod.') for k in st):
        st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    return st


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--crisis', required=True)
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    sb, sc = state_of(a.base), state_of(a.crisis)
    assert sb.keys() == sc.keys(), "checkpoint key mismatch"
    merged, n_interp, max_delta = {}, 0, 0.0
    for k in sb:
        if ('running_mean' in k or 'running_var' in k
                or 'num_batches_tracked' in k):
            assert torch.equal(sb[k].float(), sc[k].float()), \
                f"BN stat differs between base and crisis: {k} — was the " \
                f"fine-tune run with frozen_bn?"
            merged[k] = sb[k]
            continue
        if not torch.is_floating_point(sb[k]):
            merged[k] = sb[k]
            continue
        d = sc[k].float() - sb[k].float()
        merged[k] = (sb[k].float() + a.alpha * d).to(sb[k].dtype)
        n_interp += 1
        max_delta = max(max_delta, float(d.abs().max()))
    torch.save({'model': merged, 'policy_only': True,
                'args': {'merge_alpha': a.alpha, 'base': a.base,
                         'crisis': a.crisis}}, a.out)
    print(f"wrote {a.out}: alpha={a.alpha}, {n_interp} tensors interpolated, "
          f"max |task-vector| element={max_delta:.4f}")


if __name__ == '__main__':
    main()
