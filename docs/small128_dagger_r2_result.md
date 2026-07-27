# R2 arm-1 result (follow-up to postmortem review): task-vector NO-GO at the gate

Your preferred arm ran exactly as specified. Result + the one new decisive fact, then three questions.

## Setup (as you specified)

Corpus: the 25,116 judged-domain rows (recovery/prevention, teacher gap ≥1.0 — the +2.4..+2.8pp row-validated stratum), by-seed split, 15% held out. Corrections-only fine-tune of vh1, frozen BN (verified bit-identical), pairwise hinge relu(0.15 − (logit[teacher_mv] − logit[vh1_mv])), lr 1e-4, 20 epochs. Merges θ_vh1 + α·Δ for α={0.05, 0.1, 0.2, 0.4}. Pre-registered gate in deployment fp16: adoption ≥ +10pp, quiet-state drift ≤ 3%, third ≤ adoption, ≥2 useful flips per collateral flip.

## The decisive training fact

Train pref (logit_t > logit_s) climbed 0.11 → 0.77. **Held pref plateaued at ~0.49 from epoch 5 onward.** The validated corrections do not generalize across source seeds — the network learns them as ~lookup entries. Argmax adoption stayed ~0.08 even on train rows (the hinge lifts teacher-over-vh1 without making the teacher move top-1 overall).

## Gate table (3,799 held-out corrections; 18,321 quiet holdout states; all fp16)

| model | adopt% | pref% | margin med | quiet drift% | third% |
|---|---|---|---|---|---|
| vh1 | 0.5 | 0.0 | −1.062 | 0.0 | 0.4 |
| α=0.05 | 3.7 | 3.6 | −0.953 | 3.8 | 5.0 |
| α=0.10 | 6.8 | 7.5 | −0.875 | 8.2 | 10.6 |
| α=0.20 | 10.9 | 13.7 | −0.719 | 18.2 | 19.4 |
| α=0.40 | 15.7 | 25.9 | −0.438 | 40.5 | 35.6 |
| ft (α=1) | 7.4 | 50.4 | +0.008 | 89.6 | 85.6 |

Collateral ≥ useful at every α (bars wanted 2:1 the other way). Adoption is non-monotonic in α (raw ft scrambles its own argmax layer). No gameplay evals were run — the gate stopped it, as designed.

## Where this leaves the mechanism

Two very different loss geometries (dense soft/hard + rehearsal; corrections-only pairwise hinge + α-dilution) now hit the same wall on the same validated labels: ≤1 useful flip per collateral flip at any dose. Our reading: pillar3k's contested-state preferences are off-manifold for the 128ch feature geometry — per-state supervision cannot install them coherently — while the one channel that DID install cleanly (+4.6%, the gate-3 win) used MCTS-on-the-student's-own-net labels, i.e., on-manifold corrections that amplify the net's own latent preferences.

## Questions

1. Does the held-pref plateau (0.49) + non-monotonic adoption change your mechanism ranking? Is there any remaining per-state supervised variant you'd still try, or do you agree this channel is closed at this scale?
2. Our proposed pivot: re-arm the on-manifold channel — retrain the student's survival value head on the 2,000+ recorded on-policy games (we can generate 10-20k more overnight), calibration-gate it, then re-judge MCTS-on-vh1 with the better head (the value-function law: leaf value = effective teacher strength), and if the judge reads positive, reuse the exact gate-3 recipe that already won once. Critique this plan and its gates.
3. When do you call capacity? Multiple channels now read "uninstallable at 3M params." Is the 192ch control (same corpus, same recipe) now worth its cost as the decisive experiment, or premature while the on-manifold channel is untested at the new value-head strength?
