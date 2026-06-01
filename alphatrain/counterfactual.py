"""Counterfactual auxiliary loss for Phase 3 distillation (pillar3c).

Trains pillar3c to prefer floor-best moves at stationary boundary states.
Records source: alphatrain/data/stationary_counterfactuals_v1.pt (1000 anchors
mined from V13 selfplay; see scripts/mine_stationary_counterfactuals.py).

Design (locked 2026-05-24 after two rounds of ChatGPT peer review):
- Anchor filter: keep only records where top1 lost AND
      die_rate[top1] - die_rate[winner] >= 1/24
      OR leave_band_rate[top1] - leave_band_rate[winner] >= 2/24
  Drops noise-only winners. ~528 of 1000 anchors pass.
- Per-pair (winner, loser) filter: a "clean other loser" passes if
      die_rate[loser] - die_rate[winner] >= 1/24
      OR leave_band_rate[loser] - leave_band_rate[winner] >= 2/24
  Prevents listwise from re-importing R=24 sampling noise.
- Listwise margin loss:
      pair_loss = relu(margin - (logit[winner] - logit[loser]))
      anchor_loss = top1_weight * pair_loss(top1)
                  + other_weight * sum over clean losers of pair_loss(c)
      total = mean over anchors
  Weights default to 1.0 (stored top1) / 0.5 (clean others). Margin 0.25.
- Canonical orientation only — no dihedral, no color perm. N is the
  bottleneck; augmentation gives only modest variance reduction on this
  scale and adds index-transform complexity.
- Reference top1 = stored mining-time top1 (rank=1 by prior), not live
  argmax. Otherwise the target drifts during training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_DIE_THRESH = 1.0 / 24.0
DEFAULT_LEAVE_THRESH = 2.0 / 24.0
DEFAULT_MARGIN = 0.25
DEFAULT_TOP1_WEIGHT = 1.0
DEFAULT_OTHER_WEIGHT = 0.5
MAX_LOSERS_PER_ANCHOR = 8

BOARD_SIZE = 9
NUM_MOVES = 6561


def _move_to_index(move):
    """Convert ((sr, sc), (dr, dc)) → flat 6561-space index."""
    (sr, sc), (dr, dc) = move
    return (sr * BOARD_SIZE + sc) * 81 + (dr * BOARD_SIZE + dc)


def build_corpus(records_path,
                 device='cuda',
                 top1_die_thresh=DEFAULT_DIE_THRESH,
                 top1_leave_thresh=DEFAULT_LEAVE_THRESH,
                 pair_die_thresh=DEFAULT_DIE_THRESH,
                 pair_leave_thresh=DEFAULT_LEAVE_THRESH,
                 max_losers=MAX_LOSERS_PER_ANCHOR):
    """Load Phase 2 records, apply filters, return GPU-ready tensors.

    Returns a dict on `device` with these tensors:
        boards:    (N, 9, 9) int8
        next_pos:  (N, 3, 2) int8
        next_col:  (N, 3)    int8
        n_next:    (N,)      int8
        winner_idx:    (N,)    long   — floor-winner flat move index
        top1_idx:      (N,)    long   — stored top1 flat move index
        loser_idx:     (N, K)  long   — clean OTHER losers, zero-padded
        loser_mask:    (N, K)  bool   — valid loser slots
        n_clean_losers:(N,)    long   — count of clean others per anchor

    Anchors not passing the anchor filter are dropped.
    """
    raw = torch.load(records_path, map_location='cpu', weights_only=False)
    records = raw['records']
    K = max_losers

    boards, next_pos, next_col, n_next = [], [], [], []
    winner_idx, top1_idx = [], []
    loser_idx_list, loser_mask_list, n_clean_list = [], [], []

    for r in records:
        cands = sorted(r['candidates'], key=lambda c: c['rank'])
        if len(cands) < 2:
            continue
        top1 = cands[0]
        wr = r['floor_winner_rank']
        if wr == 1:
            continue
        winner = cands[wr - 1]

        die_dt = top1['die_rate'] - winner['die_rate']
        leave_dt = top1['leave_band_rate'] - winner['leave_band_rate']
        if not (die_dt >= top1_die_thresh or leave_dt >= top1_leave_thresh):
            continue

        # Build clean-other losers (excluding top1 and winner)
        clean = []
        for c in cands:
            if c['rank'] == 1 or c['rank'] == wr:
                continue
            die_d = c['die_rate'] - winner['die_rate']
            leave_d = c['leave_band_rate'] - winner['leave_band_rate']
            if die_d >= pair_die_thresh or leave_d >= pair_leave_thresh:
                clean.append(_move_to_index(c['move']))
        n_clean = min(len(clean), K)
        clean = clean[:n_clean]
        loser_pad = clean + [0] * (K - n_clean)
        loser_mask_row = [True] * n_clean + [False] * (K - n_clean)

        a = r['anchor']
        nposs = [[0, 0], [0, 0], [0, 0]]
        ncols = [0, 0, 0]
        nbs = a['next_balls']
        for i, ((nr, nc), color) in enumerate(nbs[:3]):
            nposs[i] = [nr, nc]
            ncols[i] = color
        boards.append(a['board'])
        next_pos.append(nposs)
        next_col.append(ncols)
        n_next.append(min(len(nbs), 3))
        winner_idx.append(_move_to_index(winner['move']))
        top1_idx.append(_move_to_index(top1['move']))
        loser_idx_list.append(loser_pad)
        loser_mask_list.append(loser_mask_row)
        n_clean_list.append(n_clean)

    if not boards:
        raise ValueError(
            f"No anchors passed filter (die>={top1_die_thresh:.4f} "
            f"OR leave>={top1_leave_thresh:.4f})")

    dev = torch.device(device)
    corpus = {
        'boards': torch.tensor(boards, dtype=torch.int8, device=dev),
        'next_pos': torch.tensor(next_pos, dtype=torch.int8, device=dev),
        'next_col': torch.tensor(next_col, dtype=torch.int8, device=dev),
        'n_next': torch.tensor(n_next, dtype=torch.int8, device=dev),
        'winner_idx': torch.tensor(winner_idx, dtype=torch.long, device=dev),
        'top1_idx': torch.tensor(top1_idx, dtype=torch.long, device=dev),
        'loser_idx': torch.tensor(loser_idx_list, dtype=torch.long, device=dev),
        'loser_mask': torch.tensor(loser_mask_list, dtype=torch.bool, device=dev),
        'n_clean_losers': torch.tensor(n_clean_list, dtype=torch.long, device=dev),
    }
    return corpus


def listwise_margin_loss(logits,
                         winner_idx, top1_idx, loser_idx, loser_mask,
                         margin=DEFAULT_MARGIN,
                         top1_weight=DEFAULT_TOP1_WEIGHT,
                         other_weight=DEFAULT_OTHER_WEIGHT):
    """Per-anchor weighted sum of hinge losses, then mean over anchors.

    logits:      (B, 6561) — model output on the canonical aux obs batch
    winner_idx:  (B,) long
    top1_idx:    (B,) long
    loser_idx:   (B, K) long  — padded with 0 in unused slots
    loser_mask:  (B, K) bool

    Returns scalar loss.
    """
    win = logits.gather(1, winner_idx.unsqueeze(1)).squeeze(1)        # (B,)
    top1 = logits.gather(1, top1_idx.unsqueeze(1)).squeeze(1)         # (B,)
    others = logits.gather(1, loser_idx)                              # (B, K)

    top1_loss = top1_weight * F.relu(margin - (win - top1))
    other_loss = other_weight * F.relu(margin - (win.unsqueeze(1) - others))
    other_per_anchor = (other_loss * loser_mask.float()).sum(dim=1)

    return (top1_loss + other_per_anchor).mean()


@torch.no_grad()
def preflight_metrics(logits,
                      winner_idx, top1_idx, loser_idx, loser_mask,
                      margin=DEFAULT_MARGIN):
    """Compute Phase 3 decision-gate metrics.

    Returns dict:
        stored_top1_flip_rate: fraction of anchors where logit[winner] > logit[stored_top1]
        all_clean_loser_margin_rate: fraction of all (winner, loser) pairs
            satisfying logit[winner] - logit[loser] > margin (pairs include
            both the stored-top1 pair and clean-other-loser pairs)
        n_pairs: total pair count (sanity)
    """
    win = logits.gather(1, winner_idx.unsqueeze(1)).squeeze(1)
    top1 = logits.gather(1, top1_idx.unsqueeze(1)).squeeze(1)
    others = logits.gather(1, loser_idx)

    flip = (win > top1).float().mean()

    top1_ok = (win - top1) > margin                                   # (B,)
    other_ok = ((win.unsqueeze(1) - others) > margin) & loser_mask
    n_ok = top1_ok.sum() + other_ok.sum()
    n_pairs = logits.size(0) + loser_mask.sum()

    return {
        'stored_top1_flip_rate': float(flip.item()),
        'all_clean_loser_margin_rate': float((n_ok / n_pairs.clamp(min=1)).item()),
        'n_pairs': int(n_pairs.item()),
    }
