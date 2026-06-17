"""Target audit for the completed-Q distillation target (the pre-train gate).

Both reviewers (Gemini + ChatGPT) require auditing the TARGET before any training run,
and two design questions are resolvable only from data:
  * Q4 (full-softmax vs candidate-restricted CE): measure top-10 prior MASS. If the clean
    prior already puts ~all its mass on the stored top-10, full-softmax-with-zero-target
    is safe (it barely touches off-candidate mass); else candidate-restricted is safer.
  * tau: the advantage term adv/tau must overcome the PRIOR LOGIT GAP to flip a
    correction, not just match the ~0.05 Q spread. Measure prior_gap on correction states
    and pick tau so most corrections actually flip — without collapsing onto low-visit
    moves (the raw-Q noise trap) or over-peaking.

    PYTHONPATH=. python scripts/audit_gumbel_target.py
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from alphatrain.gumbel import completed_q_target, NEG_INF

TENSOR = 'alphatrain/data/v15_pillar3f_slim.pt'


def pct(x, ps=(10, 25, 50, 75, 90, 99)):
    return {f'P{p}': round(float(torch.quantile(x.float(), p / 100)), 4) for p in ps}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--tensor', default=TENSOR)
    a.add_argument('--sample', type=int, default=800_000)
    a.add_argument('--visit-floor', type=float, default=20.0)
    a.add_argument('--kappa', type=float, default=15.0)
    a.add_argument('--spread-gate', type=float, default=0.05)
    a.add_argument('--taus', type=float, nargs='+', default=[0.01, 0.02, 0.025, 0.05])
    args = a.parse_args()

    dev = ('cuda' if torch.cuda.is_available()
           else 'mps' if torch.backends.mps.is_available() else 'cpu')
    d = torch.load(args.tensor, weights_only=True)
    N = d['cand_idx'].shape[0]
    idx = torch.randperm(N)[:args.sample]
    cv = d['cand_visit'][idx].to(dev)
    cp = d['cand_prior'][idx].to(dev)
    cq = d['cand_q'][idx].to(dev)
    nnz = d['cand_nnz'][idx].to(dev)
    rv = d['root_value'][idx].to(dev)
    cidx = d['cand_idx'][idx].to(dev)
    B, K = cv.shape
    ar = torch.arange(K, device=dev).unsqueeze(0)
    valid = ar < nnz.unsqueeze(1)
    print(f"audited {B:,} states (dev={dev})\n")

    # ── Q4: top-10 prior mass coverage ──
    # cand_prior is log clean prob; exp().sum() over valid = fraction of policy mass on top-10.
    mass = torch.where(valid, cp.exp(), torch.zeros_like(cp)).sum(1)
    print("=== Q4: top-10 prior MASS coverage (decides full vs candidate-restricted CE) ===")
    print(f"  {pct(mass)}")
    print(f"  mean {float(mass.mean()):.4f} | frac states with >=0.99 mass on top-10: "
          f"{100*float((mass>=0.99).float().mean()):.1f}% | >=0.95: "
          f"{100*float((mass>=0.95).float().mean()):.1f}%")
    print("  -> if mass is ~0.99+, full-softmax-with-zero-target is SAFE (Gemini); "
          "else candidate-restricted (ChatGPT)\n")

    # ── correction set + the prior gap that tau must overcome ──
    completed_q = (cv * cq + args.kappa * rv.unsqueeze(1)) / (cv + args.kappa)
    adv = completed_q - rv.unsqueeze(1)
    well = valid & (cv >= args.visit_floor)
    cq_well = torch.where(well, completed_q, torch.full_like(cq, NEG_INF))
    p_well = torch.where(well, cp, torch.full_like(cp, NEG_INF))
    q_arg = cq_well.argmax(1); p_arg = p_well.argmax(1)
    rows = torch.arange(B, device=dev)
    _, _, weight, is_corr, _ = completed_q_target(
        cv, cp, cq, nnz, rv, visit_floor=args.visit_floor, kappa=args.kappa,
        spread_gate=args.spread_gate, tau=args.taus[0])
    nc = int(is_corr.sum())
    print(f"=== correction set: {nc:,}/{B:,} = {100*nc/B:.2f}% ===")
    if nc == 0:
        print("no corrections in sample — increase --sample"); return
    ci = is_corr
    prior_gap = cp[rows, p_arg] - cp[rows, q_arg]            # logit favor for prior's pick
    adv_gap = adv[rows, q_arg] - adv[rows, p_arg]            # Q advantage of the better move
    tau_needed = (adv_gap / prior_gap.clamp(min=1e-6))       # tau below this flips the argmax
    print(f"  prior_logit_gap (prior pick vs Q pick) on corrections: {pct(prior_gap[ci])}")
    print(f"  shrunk adv_gap (Q pick - prior pick) on corrections:   {pct(adv_gap[ci])}")
    print(f"  tau_needed to flip (=adv_gap/prior_gap):               {pct(tau_needed[ci])}")
    print("  -> pick tau at/below the P75 of tau_needed so >=75% of corrections flip\n")

    # ── tau sweep: flip rate, low-visit collapse, peakedness, KL ──
    print("=== tau sweep (on correction states) ===")
    print(f"  {'tau':>6} | flip→Qbest | collapse<floor | mean max-p | mean KL(target‖prior) "
          f"| broad KL")
    low_visit = valid & (cv < args.visit_floor)
    for tau in args.taus:
        tp, pp, w, corr, _ = completed_q_target(
            cv, cp, cq, nnz, rv, visit_floor=args.visit_floor, kappa=args.kappa,
            spread_gate=args.spread_gate, tau=tau)
        targ_arg = tp.argmax(1)
        flip = (targ_arg[corr] == q_arg[corr]).float().mean()
        # collapse: target argmax lands on a low-visit candidate slot
        coll = low_visit[rows, targ_arg][corr].float().mean()
        maxp = tp[corr].max(1).values.mean()
        # KL(target || prior) over candidate slots
        lp_t = torch.log(tp.clamp_min(1e-9)); lp_p = torch.log(pp.clamp_min(1e-9))
        kl = (tp * (lp_t - lp_p)).sum(1)
        kl_corr = kl[corr].mean(); kl_broad = kl[~corr].mean()
        print(f"  {tau:>6.3f} | {100*float(flip):8.1f}% | {100*float(coll):11.2f}% "
              f"| {float(maxp):9.3f} | {float(kl_corr):17.3f} | {float(kl_broad):.5f}")
    print("\n  want: flip→Qbest high, collapse≈0%, max-p not ~1.0 (avoid one-hot "
          "over-commit), broad KL≈0 (the 96% untouched).")


if __name__ == '__main__':
    main()
