"""GPU-resident game engine for batched policy evaluation (protocol v2).

Wraps the validated batched_engine_gpu primitives into a full eval-grade transition with
the EXACT ColorLinesGame semantics (docs/gpu_eval_engine_plan.md §"Exact transition
semantics") that apply_move_t lacked:
  * SCORING: move-clear scores n*(n-4) on the deduped total; each spawn-clear scores
    independently per landed ball.
  * SPAWN ORDERING: all (up to) 3 balls are PLACED first (displacement sees prior
    placements but NOT prior clears, matching ColorLinesGame._spawn_balls), and only then
    are the landed cells clear-checked sequentially (a cell emptied by an earlier
    spawn-clear is skipped — clear_lines_at_t's color>0 guard).
  * turns / score / alive bookkeeping per game.

Two randomness modes:
  * INJECTED (golden tests): displacement cells and next-ball regeneration are supplied
    from a recorded CPU game — the engine is then fully deterministic and must reproduce
    ColorLinesGame bit-exactly (scripts/test_gpu_engine_golden.py).
  * SPLITMIX64 (production): per-game int64 streams seeded from the game seed. Spawn
    decisions are deterministic per seed and INDEPENDENT of batch composition (stronger
    than the CPU protocol, whose PCG64 games are reproducible only per (list, batch)).

NEW FILE — the live CPU pipelines (eval_policy, miner, crisis_mining) are untouched.
"""
import torch

from alphatrain.batched_engine_gpu import (
    BOARD, NUM_COLORS, BALLS_PER_TURN, clear_lines_at_t,
    label_components_sv, legal_priors_t, build_observation_t,
)


def choose_moves(boards, logits, labels=None):
    """Greedy argmax-legal player on device (the eval_policy policy).

    legal_priors_t with top_k=1 IS the masked argmax: top-1 logit over the full
    [B,6561] legality mask (src occupied, tgt empty, tgt component adjacent to src).
    Returns (move [B] flat int64, -1 where no legal move; has_legal [B] bool).
    """
    cnt, idx, _ = legal_priors_t(boards, logits, top_k=1, labels=labels)
    has = cnt > 0
    return torch.where(has, idx[:, 0], torch.full_like(idx[:, 0], -1)), has

NCELL = BOARD * BOARD

# SplitMix64 constants (Steele et al.) — exact int64 arithmetic; torch int64 wraps on
# overflow like uint64 two's-complement, which is what splitmix needs.
_GOLDEN = -7046029254386353131            # 0x9E3779B97F4A7C15 as signed int64
_MIX1 = -4658895280553007687              # 0xBF58476D1CE4E5B9
_MIX2 = -7723592293110705685              # 0x94D049BB133111EB


def seed_stream(seed):
    """Per-game seed value (python-int safe: wraps mod 2^64, signed int64)."""
    v = (seed * 0x9E3779B97F4A7C15 + 1) & 0xFFFFFFFFFFFFFFFF
    return v - (1 << 64) if v >= (1 << 63) else v


def _mix64(z):
    """SplitMix64 finalizer on int64 tensors (logical shifts via masking)."""
    z = z.clone()
    z ^= (z >> 30) & 0x3FFFFFFFFFFFFFFF
    z *= _MIX1
    z ^= (z >> 27) & 0x1FFFFFFFFFFFFFFF
    z *= _MIX2
    z ^= (z >> 31) & 0x1FFFFFFFFFFFFFFF
    return z


def _rand_keys_cells(seed, turns, purpose, dev):
    """[B, 81] pseudo-random int64 keys, STATELESS: key = mix(seed, turn, purpose, cell).

    Counter-based (no stream state to advance), so a game's draws depend ONLY on its
    own (seed, turn, purpose) — batch-composition independence by construction. The
    original stateful SplitMix64 streams FAILED golden test 4: any batchmate's draw
    advanced every game's stream. purpose: 0..2 displacement of spawn ball i,
    3 regen cells, 4 regen colors.

    Ranking empty cells by key gives exact uniform subsets (top-k of iid keys).
    """
    pc = ((purpose + 1) * _MIX2) & 0xFFFFFFFFFFFFFFFF      # wrap the pure-python
    pc = pc - (1 << 64) if pc >= (1 << 63) else pc          # product to signed i64
    base = _mix64(seed * _MIX1 + turns * _GOLDEN + pc)      # [B]
    cells = torch.arange(NCELL, device=dev, dtype=torch.int64)
    return _mix64(base.unsqueeze(1) + cells.unsqueeze(0) * _GOLDEN)         # [B,81]


class GpuGames:
    """B games resident on `device`. int8 boards, int64 score/turns/rng."""

    def __init__(self, batch, device):
        self.B = batch
        self.dev = torch.device(device)
        z8 = lambda *s: torch.zeros(s, dtype=torch.int8, device=self.dev)
        self.boards = z8(batch, BOARD, BOARD)
        self.next_pos = z8(batch, BALLS_PER_TURN, 2)
        self.next_col = z8(batch, BALLS_PER_TURN)
        self.n_next = torch.zeros(batch, dtype=torch.int64, device=self.dev)
        self.score = torch.zeros(batch, dtype=torch.int64, device=self.dev)
        self.turns = torch.zeros(batch, dtype=torch.int64, device=self.dev)
        self.alive = torch.zeros(batch, dtype=torch.bool, device=self.dev)
        self.rng = torch.zeros(batch, dtype=torch.int64, device=self.dev)
        self.seed = torch.zeros(batch, dtype=torch.int64, device=self.dev)
        self._kar = torch.arange(batch, device=self.dev)

    # ── helpers ──────────────────────────────────────────────────────────

    def _empty_order_rand(self, purpose):
        """Random-key empty-cell ranking, stateless per (seed, turn, purpose).
        Returns (order [B,81] best-first among empties, n_empty [B])."""
        empty = (self.boards == 0).reshape(self.B, NCELL)
        keys = _rand_keys_cells(self.rng, self.turns, purpose, self.dev)
        keys = torch.where(empty, keys, torch.full_like(keys, torch.iinfo(torch.int64).min))
        order = keys.argsort(dim=1, descending=True)
        return order, empty.sum(dim=1)

    def _score_clear(self, n_clear, active):
        pts = n_clear.to(torch.int64) * (n_clear.to(torch.int64) - 4)
        self.score += torch.where(active & (n_clear > 0), pts,
                                  torch.zeros_like(self.score))

    # ── the transition ───────────────────────────────────────────────────

    def step(self, src_flat, tgt_flat, inj=None):
        """Apply one move per ALIVE game (src/tgt as flat 0..80 cell ids; caller
        guarantees legality, matching eval_policy's argmax-legal player).

        inj (golden-test mode): dict with per-game injected randomness:
            disp  [B, 3]  flat cell for each spawned ball IF displaced (-1 = n/a)
            regen_pos [B,3,2], regen_col [B,3], regen_n [B]  — next balls afterwards
        Returns cleared_now [B] (game_over is reflected in self.alive).
        """
        B, dev, kar = self.B, self.dev, self._kar
        act0 = self.alive.clone()
        sr, sc = src_flat // BOARD, src_flat % BOARD
        tr, tc = tgt_flat // BOARD, tgt_flat % BOARD

        color = self.boards[kar, sr, sc]
        mv = kar[act0]
        self.boards[mv, sr[act0], sc[act0]] = 0
        self.boards[mv, tr[act0], tc[act0]] = color[act0]
        self.turns += act0.to(torch.int64)

        n1 = clear_lines_at_t(self.boards, tr, tc, active=act0)
        self._score_clear(n1, act0)
        spawn = act0 & (n1 == 0)

        if bool(spawn.any()):
            # Phase A: place all balls (displacement sees prior placements, NOT clears).
            land_r = torch.full((B, BALLS_PER_TURN), -1, dtype=torch.int64, device=dev)
            land_c = torch.full((B, BALLS_PER_TURN), -1, dtype=torch.int64, device=dev)
            placed = torch.zeros(B, BALLS_PER_TURN, dtype=torch.bool, device=dev)
            for i in range(BALLS_PER_TURN):
                bact = spawn & (i < self.n_next)
                if not bool(bact.any()):
                    continue
                pr = self.next_pos[:, i, 0].long()
                pc = self.next_pos[:, i, 1].long()
                pcol = self.next_col[:, i]
                free = self.boards[kar, pr, pc] == 0
                lr, lc = pr.clone(), pc.clone()
                disp = bact & ~free
                if bool(disp.any()):
                    if inj is not None:
                        cell = inj['disp'][:, i].to(dev)
                        use = disp & (cell >= 0)
                    else:
                        order, n_emp = self._empty_order_rand(purpose=i)
                        cell = order[:, 0]
                        use = disp & (n_emp > 0)
                    lr = torch.where(use, cell // BOARD, lr)
                    lc = torch.where(use, cell % BOARD, lc)
                    disp = use
                place = (bact & free) | disp
                pk = kar[place]
                self.boards[pk, lr[place], lc[place]] = pcol[place]
                land_r[:, i] = torch.where(place, lr, land_r[:, i])
                land_c[:, i] = torch.where(place, lc, land_c[:, i])
                placed[:, i] = place
            # Phase B: clear-check landed cells IN ORDER (color>0 guard inside
            # clear_lines_at_t skips cells emptied by an earlier spawn-clear).
            for i in range(BALLS_PER_TURN):
                pact = placed[:, i]
                if not bool(pact.any()):
                    continue
                ns = clear_lines_at_t(self.boards,
                                      land_r[:, i].clamp(min=0),
                                      land_c[:, i].clamp(min=0), active=pact)
                self._score_clear(ns, pact)
            # Phase C: regenerate next balls; game over when board is full.
            empty_n = (self.boards.reshape(B, NCELL) == 0).sum(dim=1)
            if inj is not None:
                want = inj['regen_n'].to(dev)
                self.next_pos = torch.where(spawn.view(B, 1, 1),
                                            inj['regen_pos'].to(dev), self.next_pos)
                self.next_col = torch.where(spawn.view(B, 1),
                                            inj['regen_col'].to(dev), self.next_col)
                self.n_next = torch.where(spawn, want, self.n_next)
            else:
                order, n_emp = self._empty_order_rand(purpose=3)
                want = torch.minimum(
                    torch.full_like(n_emp, BALLS_PER_TURN), n_emp)
                colkeys = _rand_keys_cells(self.rng, self.turns, 4,
                                           dev)[:, :BALLS_PER_TURN]
                cols = (colkeys.abs() % NUM_COLORS + 1).to(torch.int8)
                for i in range(BALLS_PER_TURN):
                    sel = spawn & (i < want)
                    cell = order[:, i]
                    self.next_pos[:, i, 0] = torch.where(
                        sel, (cell // BOARD).to(torch.int8), self.next_pos[:, i, 0])
                    self.next_pos[:, i, 1] = torch.where(
                        sel, (cell % BOARD).to(torch.int8), self.next_pos[:, i, 1])
                    self.next_col[:, i] = torch.where(
                        sel, cols[:, i], self.next_col[:, i])
                self.n_next = torch.where(spawn, want, self.n_next)
            dead = spawn & (empty_n == 0)
            self.alive &= ~dead
        return n1
