"""Completed-Q improvement target for self-play distillation (the Gumbel trunk).

WHY THIS EXISTS (see docs/gumbel_target_spec_for_review.md + project_selfplay_gumbel_recipe):
visit-distillation regressed because at 400 sims with a confident prior the visit counts
collapse to a smoothed prior (prior-domination). The fix is to distill an IMPROVEMENT
target built from the search's root Q, NOT the visit counts. But the diagnostic
(scripts/diag_q_signal.py) proved RAW Q-argmax is a noise trap: the argmax-Q candidate is
a barely-visited (median 0 visits) high-variance leaf eval 91% of the time. So Q must be
*completed* (low-visit candidates fall back to the network value) and the correction
signal must be gated to well-visited candidates with real value spread (the ~4% of states
that actually carry signal).

This module is the ONE place the review-tunable target math lives. Everything else
(dataset wiring, trainer loop, eval) is recipe-independent scaffolding. Pure torch, batched,
device-agnostic, no scatter/NUM_MOVES here — the caller scatters the per-candidate weights
into the dense move space and augments them exactly like the visit-policy target.

Per-candidate inputs are the stored top-K (padded to K with zeros; `cand_nnz` gives the
real count):
    cand_visit  (B,K) float   visit counts from the clean (no-Dirichlet) root search
    cand_prior  (B,K) float   log clean prior (pre-Dirichlet); 0.0 in padded slots
    cand_q      (B,K) float   root Q = Σvalue_sum/Σvisit_count; 0.0 in padded slots
    cand_nnz    (B,)  long     number of real candidates
    root_value  (B,)  float    network value at the root (the completion fallback)
"""
import torch
import torch.nn.functional as F

NEG_INF = -1e9


def completed_q_target(cand_visit, cand_prior, cand_q, cand_nnz, root_value, *,
                       visit_floor=20.0, tau=0.02, gamma=10.0, spread_gate=0.05,
                       kappa=15.0):
    """Build the completed-Q improvement target + anchor prior + per-state weight.

    Returns (target_p, prior_p, weight, is_correction, support), over the K candidate slots:
        target_p (B,K)  softmax(cand_prior + gated_adv/tau) over the support
        prior_p  (B,K)  softmax(cand_prior) over valid candidates  (anchor reference)
        weight   (B,)   1 + gamma * is_correction   (disagreement upweight)
        is_correction (B,) bool  trustworthy correction per the diagnostic's definition:
            >=2 well-visited candidates, real value spread, and argmax(Q) != argmax(prior)
        support  (B,K) bool  candidate-restricted CE support: well-visited on correction
            states, all-valid otherwise (off-support candidates excluded from the target)

    The math (peer-reviewed spec, Gemini + ChatGPT 2026-06-16):
      completed_Q(a) = (N(a)*cand_q[a] + kappa*root_value) / (N(a) + kappa)
                       # Bayesian shrinkage toward the network value: low-visit Q is
                       # pulled to root_value (adv->0), high-visit Q is preserved. This
                       # replaces the brittle hard visit floor and defuses the raw-Q
                       # noise trap continuously (diag_q_signal: argmax-Q was a median-0-
                       # visit leaf spike 91% of the time).
      adv(a)         = completed_Q(a) - root_value                 (center; offset cancels)
      gated_adv      = adv  on correction states, else 0           (leave the saturated 95%)
      target(a)      ∝ exp(cand_prior[a] + gated_adv(a)/tau)       over valid candidates
        NB: tau must overcome the PRIOR LOGIT GAP, not just the ~0.05 Q spread, to flip a
        correction (both reviewers). Default tau=0.02; set from the measured prior gap
        (scripts/audit_gumbel_target.py). visit_floor is used ONLY for the correction-
        eligibility flag (confidence), not for the target (shrinkage handles that).
    Padded slots are masked to -inf before every softmax/argmax so they get zero mass.
    """
    B, K = cand_visit.shape
    ar = torch.arange(K, device=cand_visit.device).unsqueeze(0)          # (1,K)
    valid = ar < cand_nnz.unsqueeze(1)                                    # (B,K) real candidate?
    rootv = root_value.unsqueeze(1)                                       # (B,1)
    neg = torch.full_like(cand_prior, NEG_INF)

    # Value estimate per candidate: light Bayesian shrinkage by visit count. NOTE we do
    # NOT use the resulting absolute level as the advantage baseline — audit_gumbel_target
    # showed root_value is systematically ABOVE every candidate Q on correction states
    # (it's the saturated network root estimate; afterstate Qs sit below it post-spawn),
    # so `adv = Q - root_value` is uniformly negative and, combined with the well-only
    # boost, sinks the explored moves below unexplored ones (low-visit collapse). Instead
    # the advantage is taken RELATIVE TO THE WELL-VISITED MAX below.
    completed_q = (cand_visit * cand_q + kappa * rootv) / (cand_visit + kappa)

    # --- per-state "trustworthy correction" flag (the diagnostic's ~4%) ---
    # Only well-explored candidates (>= visit_floor) may vote / be targets — a single
    # barely-visited leaf must not flip the policy.
    well = valid & (cand_visit >= visit_floor)
    n_well = well.sum(dim=1)                                             # (B,)
    cq_well = torch.where(well, completed_q, neg)
    p_well = torch.where(well, cand_prior, neg)
    q_arg = cq_well.argmax(dim=1)
    p_arg = p_well.argmax(dim=1)
    maxq_well = cq_well.max(dim=1, keepdim=True).values                 # (B,1)
    cq_well_min = torch.where(well, completed_q, torch.full_like(completed_q, float('inf')))
    spread = torch.where(n_well >= 1, maxq_well.squeeze(1) - cq_well_min.min(dim=1).values,
                         torch.zeros(B, device=cand_visit.device))
    is_correction = (n_well >= 2) & (q_arg != p_arg) & (spread >= spread_gate)

    # advantage RELATIVE TO THE WELL-VISITED MAX: the Q-best well-visited move gets 0,
    # every other well-visited move is pushed down by its value deficit. Unexplored
    # candidates are dropped from the target entirely (cannot be a distillation target).
    adv = completed_q - maxq_well                                       # (B,K), <=0 on well
    on_corr = is_correction.unsqueeze(1)
    # support: correction states distill over well-visited only; others over all valid.
    support = torch.where(on_corr, well, valid)
    boost = torch.where(on_corr, adv / tau, torch.zeros_like(adv))
    target_logits = torch.where(support, cand_prior + boost, neg)
    prior_logits = torch.where(valid, cand_prior, neg)
    target_p = F.softmax(target_logits, dim=1)
    prior_p = F.softmax(prior_logits, dim=1)

    weight = 1.0 + gamma * is_correction.float()
    return target_p, prior_p, weight, is_correction, support


def vetted_override_target(cand_visit, cand_prior, cand_nnz, *, min_margin=0.0):
    """DECISION distillation (the "vetted override" experiment, 3i_a).

    Treat the search as a vetting process: if the search's most-visited move (its actual
    decision) equals the prior's argmax, the prior is already good enough at this sim
    budget — skip the state (weight 0). If the search picks a DIFFERENT move, that's a
    high-SNR prior-blunder signal — hard one-hot BC toward the search's move.

    This is NOT the soft visit distribution (smearing) nor completed-Q (which distilled the
    REJECTED low-visit move). The target is the teacher's ACTUAL decision = argmax(visits).

    Returns the same 5-tuple as completed_q_target so it drops into the dataset/trainer:
        target_p (B,K)  ONE-HOT at the search's most-visited candidate (the override move)
        prior_p  (B,K)  softmax(cand_prior) over valid candidates  (anchor reference)
        weight   (B,)   1.0 on override states (argmax visits != argmax prior), else 0.0
        is_override (B,) bool
        support  (B,K) bool  valid candidates (candidate-restricted CE support)

    Agreement states get weight 0; the trainer's `--beta` adds an optional tiny CE-to-prior
    anchor on ALL states (prevents the backbone from drifting on the 84% it already gets
    right). beta=0 = "ignore agreement" (drop); beta>0 = "tiny KL-anchor to prior".
    `min_margin`>0 additionally requires the override move to beat the prior's move by that
    visit-share margin (only count CONFIDENT overrides, not flat-distribution coin-flips).
    """
    B, K = cand_visit.shape
    ar = torch.arange(K, device=cand_visit.device).unsqueeze(0)
    valid = ar < cand_nnz.unsqueeze(1)
    neg = torch.full_like(cand_prior, NEG_INF)
    v_masked = torch.where(valid, cand_visit, torch.full_like(cand_visit, NEG_INF))
    p_masked = torch.where(valid, cand_prior, neg)
    va = v_masked.argmax(dim=1)                 # search's most-visited (played) move
    pa = p_masked.argmax(dim=1)                 # prior's top move
    override = (va != pa)
    if min_margin > 0.0:
        vis = torch.where(valid, cand_visit, torch.zeros_like(cand_visit))
        vis = vis / vis.sum(dim=1, keepdim=True).clamp_min(1e-9)
        rows = torch.arange(B, device=cand_visit.device)
        override = override & ((vis[rows, va] - vis[rows, pa]) >= min_margin)
    target_p = F.one_hot(va, num_classes=K).float()     # hard one-hot at the override move
    prior_p = F.softmax(p_masked, dim=1)
    weight = override.float()
    return target_p, prior_p, weight, override, valid
