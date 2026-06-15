"""Neural MCTS for Color Lines 98.

AlphaZero-style Monte Carlo Tree Search using the trained ResNet:
- Policy head provides move priors (which moves to explore)
- Value head evaluates leaf positions (replaces heuristic rollouts)
- PUCT selection balances exploration vs exploitation
- Virtual loss batching: collect B leaves per batch, one NN forward pass
- Determinized: each simulation samples independent ball spawns via game.clone()
- MuZero-style Q normalization for score-based (not win/loss) values

Performance optimizations:
- Flat integer action keys (avoids tuple-of-tuples overhead)
- Shared RNG across simulations (avoids 5us np.random.default_rng per clone)
- Inlined PUCT selection (avoids method call + property overhead)
- Direct _legal_priors_jit → dict with int keys (no numpy decomposition)

Nodes represent post-spawn game states — matching the NN's training
distribution. 400 simulations average over spawn outcomes.

Usage:
    python -m alphatrain.evaluate --player mcts --simulations 400 --games 20
"""

import math
import numpy as np
import torch
from numba import njit

from alphatrain.observation import build_observation
from alphatrain.scripts.mine_death_features import (
    board_features, board_features_with_next,
)
from game.board import _label_empty_components, _is_reachable

BOARD_SIZE = 9
NUM_MOVES = BOARD_SIZE ** 4  # 6561

VIRTUAL_LOSS = 1.0


@njit(cache=True)
def _evaluate_board(board):
    """Fast board evaluation for MCTS leaf nodes.

    Returns a score where higher = healthier board.
    Captures: empty count, largest connected region, line potential,
    and partition penalty.
    """
    empty = 0
    max_region = 0
    n_regions = 0
    visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    queue_r = np.empty(81, dtype=np.int8)
    queue_c = np.empty(81, dtype=np.int8)

    # BFS to find connected empty regions
    for sr in range(BOARD_SIZE):
        for sc in range(BOARD_SIZE):
            if board[sr, sc] == 0:
                empty += 1
                if visited[sr, sc] == 0:
                    # Flood fill this region
                    n_regions += 1
                    region_size = 0
                    visited[sr, sc] = 1
                    queue_r[0] = sr
                    queue_c[0] = sc
                    head, tail = 0, 1
                    while head < tail:
                        r, c = queue_r[head], queue_c[head]
                        head += 1
                        region_size += 1
                        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                                if board[nr, nc] == 0 and visited[nr, nc] == 0:
                                    visited[nr, nc] = 1
                                    queue_r[tail] = nr
                                    queue_c[tail] = nc
                                    tail += 1
                    if region_size > max_region:
                        max_region = region_size

    # Line potential: count balls in partial lines (3-4 in a row)
    line_potential = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            color = board[r, c]
            if color == 0:
                continue
            # Check horizontal and vertical only (avoid double-counting)
            for dr, dc in ((0, 1), (1, 0)):
                length = 1
                cr, cc = r + dr, c + dc
                while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == color:
                    length += 1
                    cr += dr
                    cc += dc
                if 3 <= length <= 4:
                    line_potential += length

    # Score: empty space + connectivity bonus + line potential - partition penalty
    # Partition: if the board has multiple disconnected empty regions,
    # balls can't move freely between them
    partition_penalty = 0
    if n_regions > 1:
        # Penalty proportional to how fragmented the board is
        # (empty - max_region) = cells in non-largest regions
        partition_penalty = (empty - max_region) * 2

    return float(empty + max_region + line_potential - partition_penalty)


@njit(cache=True)
def _evaluate_features_linear(board, next_r, next_c, next_col, n_next,
                              coefs, means, stds, bias):
    """Linear value function over 27 board+next-ball features.

    V(board, next) = bias + Σ_i coefs[i] · (feat_i − means[i]) / stds[i].
    Feature order: 16 board features (FEATURE_NAMES) + 9 next-ball features
    (NEXT_BALL_FEATURE_NAMES) + 2 derived [ratio, frag_score] = 27 total.

    Next-ball features include aggregate deltas, tactical line completion
    indicators (max_next_line, next_makes_4plus, next_makes_5plus), and
    the cheap landing-cell stats. The threshold-encoded indicators let
    the linear model learn the step function "+1 from going 3→4 length"
    that a single continuous feature can't represent on its own.

    Returns float scalar suitable for use as MCTS leaf value.
    """
    feats = board_features_with_next(board, next_r, next_c, next_col, n_next)
    empty = feats[0]
    n_components = feats[1]
    largest = feats[2]
    denom_empty = empty if empty > 0 else 1
    ratio = largest / denom_empty
    frag_score = (empty - largest) * n_components

    v = bias
    # 16 board features
    v += coefs[0] * (feats[0] - means[0]) / stds[0]
    v += coefs[1] * (feats[1] - means[1]) / stds[1]
    v += coefs[2] * (feats[2] - means[2]) / stds[2]
    v += coefs[3] * (feats[3] - means[3]) / stds[3]
    v += coefs[4] * (feats[4] - means[4]) / stds[4]
    v += coefs[5] * (feats[5] - means[5]) / stds[5]
    v += coefs[6] * (feats[6] - means[6]) / stds[6]
    v += coefs[7] * (feats[7] - means[7]) / stds[7]
    v += coefs[8] * (feats[8] - means[8]) / stds[8]
    v += coefs[9] * (feats[9] - means[9]) / stds[9]
    v += coefs[10] * (feats[10] - means[10]) / stds[10]
    v += coefs[11] * (feats[11] - means[11]) / stds[11]
    v += coefs[12] * (feats[12] - means[12]) / stds[12]
    v += coefs[13] * (feats[13] - means[13]) / stds[13]
    v += coefs[14] * (feats[14] - means[14]) / stds[14]
    v += coefs[15] * (feats[15] - means[15]) / stds[15]
    # 9 next-ball features
    v += coefs[16] * (feats[16] - means[16]) / stds[16]
    v += coefs[17] * (feats[17] - means[17]) / stds[17]
    v += coefs[18] * (feats[18] - means[18]) / stds[18]
    v += coefs[19] * (feats[19] - means[19]) / stds[19]
    v += coefs[20] * (feats[20] - means[20]) / stds[20]
    v += coefs[21] * (feats[21] - means[21]) / stds[21]
    v += coefs[22] * (feats[22] - means[22]) / stds[22]
    v += coefs[23] * (feats[23] - means[23]) / stds[23]
    v += coefs[24] * (feats[24] - means[24]) / stds[24]
    # 2 derived
    v += coefs[25] * (ratio - means[25]) / stds[25]
    v += coefs[26] * (frag_score - means[26]) / stds[26]
    return v


@njit(cache=True)
def _legal_priors_jit(board, pol_logits, top_k):
    """Compute top-K legal move priors entirely in JIT.

    Returns (count, flat_indices[top_k], priors[top_k]).

    Optimized: BFS labels + precomputed per-component cell lists so each
    source ball iterates only its reachable cells (not all 81). Top-K is
    maintained inline via a sorted insertion buffer, and softmax is computed
    over only the final K values instead of all legal moves.
    """
    # Label connected components of empty cells and collect per-component
    # cell lists in a single BFS pass
    labels = np.zeros((9, 9), dtype=np.int8)
    queue = np.empty(162, dtype=np.int32)
    current = np.int8(0)
    comp_start = np.empty(82, dtype=np.int32)
    comp_count = np.empty(82, dtype=np.int32)
    comp_flat = np.empty(81, dtype=np.int32)
    comp_count[:] = 0
    n_empty = 0

    for sr in range(9):
        for sc in range(9):
            if board[sr, sc] != 0 or labels[sr, sc] != 0:
                continue
            current += 1
            labels[sr, sc] = current
            comp_start[current] = n_empty
            queue[0] = sr * 9 + sc
            head = 0
            tail = 1
            comp_flat[n_empty] = sr * 9 + sc
            n_empty += 1
            while head < tail:
                pos = queue[head]
                head += 1
                r = pos // 9
                c = pos % 9
                for d in range(4):
                    if d == 0:
                        nr, nc = r, c + 1
                    elif d == 1:
                        nr, nc = r, c - 1
                    elif d == 2:
                        nr, nc = r + 1, c
                    else:
                        nr, nc = r - 1, c
                    if 0 <= nr < 9 and 0 <= nc < 9:
                        if board[nr, nc] == 0 and labels[nr, nc] == 0:
                            labels[nr, nc] = current
                            queue[tail] = nr * 9 + nc
                            tail += 1
                            comp_flat[n_empty] = nr * 9 + nc
                            n_empty += 1
            comp_count[current] = n_empty - comp_start[current]

    # Collect legal moves with inline top-K maintenance
    # Instead of collecting ALL moves then sorting, we maintain a sorted buffer
    # of the K best logits. Softmax is deferred to only these K values.
    k = top_k
    topk_idx = np.empty(k, dtype=np.int32)
    topk_log = np.empty(k, dtype=np.float32)
    n_top = 0
    min_top = np.float32(-1e30)
    n_moves = 0

    for sr in range(9):
        for sc in range(9):
            if board[sr, sc] == 0:
                continue
            # Find reachable component IDs (deduplicated)
            adj_comp = np.empty(4, dtype=np.int8)
            n_adj = 0
            for d in range(4):
                if d == 0:
                    nr, nc = sr, sc + 1
                elif d == 1:
                    nr, nc = sr, sc - 1
                elif d == 2:
                    nr, nc = sr + 1, sc
                else:
                    nr, nc = sr - 1, sc
                if 0 <= nr < 9 and 0 <= nc < 9:
                    lbl = labels[nr, nc]
                    if lbl > 0:
                        found = False
                        for j in range(n_adj):
                            if adj_comp[j] == lbl:
                                found = True
                                break
                        if not found:
                            adj_comp[n_adj] = lbl
                            n_adj += 1

            if n_adj == 0:
                continue

            base = (sr * 9 + sc) * 81

            # Iterate only cells in reachable components
            for ci in range(n_adj):
                comp_id = adj_comp[ci]
                start = comp_start[comp_id]
                count = comp_count[comp_id]
                for j in range(count):
                    tgt = comp_flat[start + j]
                    idx = base + tgt
                    logit = pol_logits[idx]
                    n_moves += 1

                    # Maintain top-K sorted buffer (ascending, smallest first)
                    if n_top < k:
                        pos = n_top
                        while pos > 0 and topk_log[pos - 1] > logit:
                            topk_idx[pos] = topk_idx[pos - 1]
                            topk_log[pos] = topk_log[pos - 1]
                            pos -= 1
                        topk_idx[pos] = idx
                        topk_log[pos] = logit
                        n_top += 1
                        if n_top == k:
                            min_top = topk_log[0]
                    elif logit > min_top:
                        # Evict smallest, shift and insert
                        pos = 0
                        while pos < k - 1 and topk_log[pos + 1] < logit:
                            topk_idx[pos] = topk_idx[pos + 1]
                            topk_log[pos] = topk_log[pos + 1]
                            pos += 1
                        topk_idx[pos] = idx
                        topk_log[pos] = logit
                        min_top = topk_log[0]

    if n_moves == 0:
        return 0, topk_idx[:0], topk_log[:0]

    # Softmax over just the K selected values
    actual_k = n_top
    max_l = topk_log[0]
    for i in range(1, actual_k):
        if topk_log[i] > max_l:
            max_l = topk_log[i]
    total = np.float32(0.0)
    priors = np.empty(actual_k, dtype=np.float32)
    for i in range(actual_k):
        p = np.exp(topk_log[i] - max_l)
        priors[i] = p
        total += p
    for i in range(actual_k):
        priors[i] /= total

    return actual_k, topk_idx[:actual_k], priors


def _flat_to_action(flat_idx):
    """Decode flat index to ((sr, sc), (tr, tc)) action tuple."""
    src_flat = flat_idx // 81
    tgt_flat = flat_idx % 81
    return ((src_flat // 9, src_flat % 9), (tgt_flat // 9, tgt_flat % 9))


class Node:
    """MCTS tree node."""
    __slots__ = ('children', 'visit_count', 'value_sum', 'prior')

    def __init__(self, prior=0.0):
        self.children = {}  # flat_action_int -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self):
        return len(self.children) > 0


# Pre-allocated buffers for _build_obs_for_game (avoid per-call allocation)
_obs_nr = np.zeros(3, dtype=np.intp)
_obs_nc = np.zeros(3, dtype=np.intp)
_obs_ncol = np.zeros(3, dtype=np.intp)


def _build_obs_for_game(game):
    """Build observation tensor from game state."""
    nr, nc, ncol = _obs_nr, _obs_nc, _obs_ncol
    nb = game.next_balls
    nn = len(nb)
    if nn > 3:
        nn = 3
    if nn >= 1:
        pos_col = nb[0]
        nr[0] = pos_col[0][0]; nc[0] = pos_col[0][1]; ncol[0] = pos_col[1]
    else:
        nr[0] = 0; nc[0] = 0; ncol[0] = 0
    if nn >= 2:
        pos_col = nb[1]
        nr[1] = pos_col[0][0]; nc[1] = pos_col[0][1]; ncol[1] = pos_col[1]
    else:
        nr[1] = 0; nc[1] = 0; ncol[1] = 0
    if nn >= 3:
        pos_col = nb[2]
        nr[2] = pos_col[0][0]; nc[2] = pos_col[0][1]; ncol[2] = pos_col[1]
    else:
        nr[2] = 0; nc[2] = 0; ncol[2] = 0
    return build_observation(game.board, nr, nc, ncol, nn)


def _get_legal_priors(game, pol_logits_np, top_k):
    """Extract legal move priors from policy logits, keep top-K.

    Uses JIT-compiled function for speed (computes connected components,
    legal moves, softmax, and top-K entirely in numba).

    Returns dict mapping ((sr,sc),(tr,tc)) -> prior (legacy tuple format).
    """
    k, flat_idx, priors = _legal_priors_jit(
        game.board, pol_logits_np, top_k)
    if k == 0:
        return {}
    # Vectorized index -> action conversion (avoids Python loop)
    idx = flat_idx[:k].astype(np.int32)
    src_flat = idx // 81
    tgt_flat = idx % 81
    sr = src_flat // 9
    sc = src_flat % 9
    tr = tgt_flat // 9
    tc = tgt_flat % 9
    pri = priors[:k]
    return {((int(sr[i]), int(sc[i])), (int(tr[i]), int(tc[i]))): float(pri[i])
            for i in range(k)}


def _get_legal_priors_flat(board, pol_logits_np, top_k):
    """Extract legal move priors as flat-index dict {int: float}.

    Same as _get_legal_priors but avoids tuple-of-tuples key overhead.
    Returns dict mapping flat_action_int -> prior.
    """
    k, flat_idx, priors = _legal_priors_jit(board, pol_logits_np, top_k)
    if k == 0:
        return {}
    # Direct int keys — no numpy decomposition needed
    return {int(flat_idx[i]): float(priors[i]) for i in range(k)}


class MCTS:
    """Neural MCTS with PUCT selection, virtual loss batching, and value-head
    leaf evaluation.

    Args:
        net: AlphaTrainNet model (eval mode, on device)
        device: torch device
        num_simulations: simulations per search (default 400)
        c_puct: exploration constant (default 2.5)
        top_k: max children per node (default 30)
        batch_size: leaves per batched NN eval (default 16)
    """

    def __init__(self, net=None, device=None, max_score=30000.0,
                 num_simulations=400, c_puct=2.5, top_k=30, batch_size=16,
                 inference_client=None, dynamic_sims=False,
                 heuristic_value=False, value_net=None,
                 terminal_value=None, override_threshold=0.0,
                 feature_weights_path=None, early_stop=False,
                 q_weight=1.0, value_head_path=None):
        self.net = net
        self.device = device
        self.max_score = max_score
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.top_k = top_k
        self.batch_size = batch_size
        self.inference_client = inference_client
        self.dynamic_sims = dynamic_sims
        self.heuristic_value = heuristic_value
        self.value_net = value_net
        self.terminal_value = terminal_value
        self.override_threshold = override_threshold
        self.early_stop = early_stop
        # PUCT: score = q_weight * q_norm + U. Default 1.0 (current
        # behavior). q_weight=0 makes search pure-prior (no Q signal,
        # virtual loss only). Useful for diagnosing whether the leaf
        # evaluator is contributing or just adding noise — see
        # ChatGPT review note 2026-05-05.
        self.q_weight = q_weight
        self._fp16 = False
        self._sim_rng = None  # SimpleRng, set per search

        # Pre-allocated next-ball buffers — used per leaf to feed the
        # feature evaluator. Reused across calls to avoid allocation churn.
        self._nb_r = np.zeros(3, dtype=np.int8)
        self._nb_c = np.zeros(3, dtype=np.int8)
        self._nb_col = np.zeros(3, dtype=np.int8)

        # Feature-based linear value evaluator (replaces NN value head when set)
        self.feature_coefs = None
        self.feature_means = None
        self.feature_stds = None
        self.feature_bias = 0.0
        if feature_weights_path is not None:
            data = np.load(feature_weights_path)
            self.feature_coefs = data['coefs'].astype(np.float32)
            self.feature_means = data['means'].astype(np.float32)
            self.feature_stds = data['stds'].astype(np.float32)
            self.feature_bias = float(data['bias'])
            assert self.feature_coefs.shape[0] == 27, (
                f"Expected 27 feature coefs (16 board + 9 next-ball + 2 "
                f"derived), got {self.feature_coefs.shape[0]}. The feature "
                f"format changed when next-ball features were added — refit "
                f"weights with the current fit_feature_value.py.")

        # NN value head (multi-horizon survival classifier). Mutually
        # exclusive with feature_weights_path — one Q source at a time.
        #
        # Two modes:
        #   - Local (no inference_client): MCTS owns the head and runs
        #     it over backbone features after the policy forward pass.
        #   - Server (inference_client): the server already runs the head
        #     fused with the policy net and writes scalar V into val_buf.
        #     MCTS keeps a flag so leaf eval reads val_np instead of
        #     computing V from features.
        self.value_head = None
        self.value_head_via_server = False
        self.value_head_target_type = 'survival'  # density heads use no sigmoid
        self.value_head_type = 'value_head'       # 'spatial' uses raw scalar
        if value_head_path is not None:
            if feature_weights_path is not None:
                raise ValueError(
                    "value_head_path and feature_weights_path are "
                    "mutually exclusive. Pick one.")
            from alphatrain.value_head import (
                load_any as _load_any, DEFAULT_HORIZON_WEIGHTS as _DEFAULT_W)
            if inference_client is not None:
                # Server runs the head and computes scalar V. MCTS reads
                # val_np from val_buf.
                self.value_head_via_server = True
                self._horizon_weights = torch.tensor(
                    _DEFAULT_W, dtype=torch.float32,
                    device=device if device is not None else 'cpu')
            else:
                if net is None:
                    raise ValueError(
                        "value_head_path requires net to be set "
                        "(value head reuses the policy backbone features).")
                self.value_head, ckpt, head_type = _load_any(
                    value_head_path, device=device)
                self.value_head_type = head_type
                self.value_head_target_type = ckpt.get('target_type', 'survival')
                if head_type == 'spatial':
                    # Single scalar output; no horizon weighting at inference.
                    self._horizon_weights = None
                elif self.value_head_target_type == 'density':
                    density_weights = (0.5, 0.3, 0.2)
                    self._horizon_weights = torch.tensor(
                        density_weights, dtype=torch.float32, device=device)
                else:
                    self._horizon_weights = torch.tensor(
                        _DEFAULT_W, dtype=torch.float32, device=device)

        # Pre-allocate obs buffer for server mode (reused across searches)
        if inference_client is not None:
            self._obs_np_buf = np.empty(
                (batch_size, 18, 9, 9), dtype=np.float32)
        if net is not None and device is not None:
            # Detect fp16 model (JIT traced or .half())
            try:
                p = next(net.parameters())
                self._fp16 = (p.dtype == torch.float16)
            except (StopIteration, AttributeError):
                pass
            dtype = torch.float16 if self._fp16 else torch.float32
            self._obs_buf = torch.empty(batch_size, 18, 9, 9,
                                        device=device, dtype=dtype)

    def _value_head_eval_single(self, game):
        """Run the NN value head on a single game state.

        Used for the root and for terminal boards (which aren't part of
        the batched leaf forward). Cheap (one forward) and rare per
        search.
        """
        obs_np = _build_obs_for_game(game)
        net_dtype = torch.float16 if self._fp16 else torch.float32
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(
            device=self.device, dtype=net_dtype)
        with torch.inference_mode():
            feats = self.net.backbone_features(obs)
            vh_out = self.value_head(feats.float())
            if self.value_head_type == 'spatial':
                scalar = vh_out.squeeze(-1)
            elif self.value_head_target_type == 'density':
                scalar = (vh_out * self._horizon_weights).sum(dim=-1)
            else:
                probs = torch.sigmoid(vh_out)
                scalar = (probs * self._horizon_weights).sum(dim=-1)
        return float(scalar.item())

    def _fill_next_ball_buffers(self, game):
        """Fill self._nb_r/_nb_c/_nb_col from game.next_balls (up to 3).

        Returns the count of valid next balls (0-3). Used as input to the
        feature evaluator's next-ball-aware feature extraction.
        """
        nb = game.next_balls
        n = len(nb)
        if n > 3:
            n = 3
        for i in range(n):
            pos, col = nb[i]
            self._nb_r[i] = pos[0]
            self._nb_c[i] = pos[1]
            self._nb_col[i] = col
        return n

    def _nn_evaluate_single(self, game):
        """Single NN forward pass -> (priors dict, value scalar).

        Uses inference_client if available, otherwise direct net call.
        When value_net is set, uses it for value instead of the policy
        model's value head.
        Returns priors as {flat_int: prior}.
        """
        obs_np = _build_obs_for_game(game)
        if self.inference_client is not None:
            pol_np, value = self.inference_client.evaluate(obs_np)
            priors = _get_legal_priors_flat(game.board, pol_np, self.top_k)
            # If value_net is loaded in-process, override server's value
            if self.value_net is not None:
                obs_t = torch.from_numpy(obs_np).unsqueeze(0)
                vnet_device = next(self.value_net.parameters()).device
                vnet_dtype = next(self.value_net.parameters()).dtype
                obs_t = obs_t.to(device=vnet_device, dtype=vnet_dtype)
                with torch.inference_mode():
                    vnet_logits = self.value_net(obs_t)
                    value = torch.sigmoid(vnet_logits.squeeze(-1)).item()
            return priors, value
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        if self._fp16:
            obs = obs.half()
        with torch.inference_mode():
            out = self.net(obs)
            # policy_only models return just policy_logits; dual-head
            # models return (policy_logits, value_logits).
            if isinstance(out, tuple):
                pol_logits, val_logits = out
            else:
                pol_logits, val_logits = out, None
            if self.value_net is not None:
                vnet_dtype = next(self.value_net.parameters()).dtype
                vnet_obs = obs.to(dtype=vnet_dtype)
                vnet_logits = self.value_net(vnet_obs)
                value = torch.sigmoid(vnet_logits.squeeze(-1)).item()
            elif self.feature_coefs is not None:
                # Caller will override this with feature value at root;
                # placeholder is fine.
                value = 0.0
            elif self.value_head is not None:
                # Same — caller overrides with NN value head at root.
                value = 0.0
            elif val_logits is not None:
                value = self.net.predict_value(
                    val_logits, max_val=self.max_score).item()
            else:
                raise RuntimeError(
                    "policy_only model has no NN value head; provide "
                    "feature_weights_path, value_head_path, or value_net "
                    "for MCTS leaf eval.")
        pol_np = pol_logits[0].float().cpu().numpy()
        priors = _get_legal_priors_flat(game.board, pol_np, self.top_k)
        return priors, value

    def _select_child(self, node, min_q, max_q):
        """PUCT selection with MuZero-style Q normalization."""
        best_score = float('-inf')
        best_action = None
        best_child = None
        sqrt_parent = math.sqrt(node.visit_count)
        q_range = max_q - min_q
        q_weight = self.q_weight

        for action, child in node.children.items():
            if child.visit_count > 0:
                q = child.value_sum / child.visit_count
                q_norm = (q - min_q) / q_range if q_range > 0 else 0.5
            else:
                q_norm = 0.5
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q_weight * q_norm + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def search(self, game, temperature=0.0, dirichlet_alpha=0.0,
               dirichlet_weight=0.0, return_policy=False,
               force_full_search=False):
        """Run batched MCTS with virtual loss from current game state.

        Args:
            temperature: move selection temperature (0=argmax, >0=proportional
                to visit counts). Used for self-play exploration.
            dirichlet_alpha: Dirichlet noise parameter for root priors (0=off).
            dirichlet_weight: weight for Dirichlet noise (e.g., 0.25).
            return_policy: if True, return (action, policy_target) where
                policy_target is a (6561,) array of normalized visit counts.

        Returns:
            action tuple, or (action, policy_target) if return_policy=True.
        """
        root = Node()

        # Seed sim_rng deterministically from full game state.
        # Hash full board + next_balls + score + turns for unique seed.
        import hashlib
        h = hashlib.md5(game.board.tobytes())
        for pos, color in game.next_balls:
            h.update(bytes([pos[0], pos[1], color]))
        h.update(game.score.to_bytes(4, 'little', signed=False))
        h.update(game.turns.to_bytes(4, 'little', signed=False))
        state_seed = int.from_bytes(h.digest()[:8], 'little')
        from game.rng import SimpleRng
        self._sim_rng = SimpleRng(state_seed)

        # Expand root
        priors, root_value = self._nn_evaluate_single(game)
        if not priors:
            return (None, None) if return_policy else None
        # Override NN root_value with feature evaluator if enabled, so
        # min_q/max_q anchor matches subsequent leaf values
        if self.feature_coefs is not None:
            n_next = self._fill_next_ball_buffers(game)
            root_value = float(_evaluate_features_linear(
                game.board, self._nb_r, self._nb_c, self._nb_col, n_next,
                self.feature_coefs, self.feature_means,
                self.feature_stds, self.feature_bias))
        elif self.value_head is not None:
            root_value = self._value_head_eval_single(game)
        for action, prior in priors.items():
            root.children[action] = Node(prior=prior)
        root.visit_count = 1
        root.value_sum = root_value

        # Measure policy confidence BEFORE Dirichlet noise
        # (Dirichlet reduces P_max by ~25%, masking true confidence)
        raw_max_prior = max(c.prior for c in root.children.values()) if root.children else 0.0

        # Stash the CLEAN (pre-Dirichlet) priors + root value for label recording.
        # These are the noise-free improvement-target ingredients (Gumbel softmax(z+Q)):
        # the prior here is the raw net policy, before exploration noise is mixed in.
        self._last_raw_priors = dict(priors)
        self._last_root_value = root_value

        # Dirichlet noise at root for exploration
        if dirichlet_alpha > 0 and dirichlet_weight > 0:
            noise = self._sim_rng.dirichlet(
                [dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children.values()):
                child.prior = ((1 - dirichlet_weight) * child.prior
                               + dirichlet_weight * noise[i])

        min_q = root_value
        max_q = root_value

        # Localize frequently accessed attributes for speed
        c_puct = self.c_puct
        top_k = self.top_k
        batch_size = self.batch_size
        num_sims = self.num_simulations
        q_weight = self.q_weight

        # Dynamic sims: reduce search for positions where policy is confident.
        # Check raw prior (before Dirichlet) — reflects true policy confidence.
        # With top_k=30 softmax, P_max typically ranges 0.05-0.70.
        if self.dynamic_sims and root.children and not force_full_search:
            if raw_max_prior > 0.5:
                num_sims = max(50, num_sims // 10)
            elif raw_max_prior > 0.3:
                num_sims = max(100, num_sims // 4)
        self._last_effective_sims = num_sims
        self._last_max_prior = raw_max_prior
        sim_rng = self._sim_rng
        use_server = self.inference_client is not None
        if use_server:
            obs_np_buf = self._obs_np_buf

        # Vectorized root selection. The root has top_k (~300) children and is
        # scanned on EVERY simulation's first descent step — the mining CPU hot
        # loop. These arrays MIRROR the root children's Node stats (kept in
        # lockstep in the virtual-loss + backup steps below) so the depth-0 PUCT
        # argmax becomes one numpy op instead of a ~300-iter Python loop. Deeper
        # nodes (small fan-out) keep the Python path. Bit-identical to the scalar
        # loop (same float ops, first-max argmax) — proven by golden_search_test.py.
        _root_items = list(root.children.items())
        n_root = len(_root_items)
        root_actions_arr = np.fromiter((a for a, _ in _root_items),
                                       dtype=np.int64, count=n_root)
        root_nodes_list = [c for _, c in _root_items]
        root_cpp = c_puct * np.fromiter((c.prior for _, c in _root_items),
                                        dtype=np.float64, count=n_root)
        root_vc_arr = np.zeros(n_root, dtype=np.float64)    # mirrors visit_count
        root_vsum_arr = np.zeros(n_root, dtype=np.float64)  # mirrors value_sum

        sims_done = 0
        while sims_done < num_sims:
            bs = min(batch_size, num_sims - sims_done)
            batch_paths = []
            batch_games = []
            batch_leaf_nodes = []
            batch_game_over = []
            batch_root_idx = []
            obs_count = 0

            # === SELECT B leaves with virtual loss ===
            for _ in range(bs):
                node = root
                sim_game = game.clone(rng=sim_rng)
                path = [node]
                cur_root_idx = -1

                depth = 0
                while node.children and not sim_game.game_over:
                    # Depth 0 (root, ~300 children, all valid): vectorized PUCT
                    # argmax over the mirror arrays — bit-identical to the scalar
                    # loop below, but one numpy op instead of ~300 Python iters.
                    if depth == 0:
                        sqrt_parent = math.sqrt(node.visit_count)
                        q_range = max_q - min_q
                        vc = root_vc_arr
                        if q_range > 0:
                            q_norm = (root_vsum_arr / np.where(vc > 0, vc, 1.0)
                                      - min_q) / q_range
                            np.putmask(q_norm, vc == 0, 0.5)
                        else:
                            q_norm = np.full(n_root, 0.5)
                        score = q_weight * q_norm \
                            + root_cpp * sqrt_parent / (1.0 + vc)
                        ri = int(np.argmax(score))
                        cur_root_idx = ri
                        best_child = root_nodes_list[ri]
                        best_action = int(root_actions_arr[ri])
                        src_flat = best_action // 81
                        tgt_flat = best_action % 81
                        sim_game.trusted_move(src_flat // 9, src_flat % 9,
                                              tgt_flat // 9, tgt_flat % 9)
                        path.append(best_child)
                        node = best_child
                        depth += 1
                        continue

                    # Open-loop PUCT with lazy reachability validation.
                    # Depth 1+: cheap occupancy filter on all children,
                    # then reachability check only on the chosen move.
                    sqrt_parent = math.sqrt(node.visit_count)
                    q_range = max_q - min_q
                    board = sim_game.board
                    need_filter = depth > 0
                    banned = None
                    cc_labels = None

                    while True:
                        best_score = -1e30
                        best_action = 0
                        best_child = None

                        for act_i, child in node.children.items():
                            if banned and act_i in banned:
                                continue
                            if need_filter:
                                src_f = act_i // 81
                                tgt_f = act_i % 81
                                if board[src_f // 9, src_f % 9] == 0:
                                    continue
                                if board[tgt_f // 9, tgt_f % 9] != 0:
                                    continue
                            vc = child.visit_count
                            if vc > 0:
                                q = child.value_sum / vc
                                q_norm = (q - min_q) / q_range \
                                    if q_range > 0 else 0.5
                            else:
                                q_norm = 0.5
                            u = c_puct * child.prior * sqrt_parent / (1 + vc)
                            score = q_weight * q_norm + u
                            if score > best_score:
                                best_score = score
                                best_action = act_i
                                best_child = child

                        if best_child is None:
                            break

                        if not need_filter:
                            break  # root: always valid

                        # Lazy reachability: check only the chosen move
                        if cc_labels is None:
                            cc_labels = _label_empty_components(board)
                        src_f = best_action // 81
                        tgt_f = best_action % 81
                        if _is_reachable(cc_labels,
                                         src_f // 9, src_f % 9,
                                         tgt_f // 9, tgt_f % 9):
                            break  # valid move

                        # Unreachable: ban and retry
                        if banned is None:
                            banned = set()
                        banned.add(best_action)

                    if best_child is None:
                        break  # no legal children on this board

                    src_flat = best_action // 81
                    tgt_flat = best_action % 81
                    sim_game.trusted_move(
                        src_flat // 9, src_flat % 9,
                        tgt_flat // 9, tgt_flat % 9)
                    path.append(best_child)
                    node = best_child
                    depth += 1

                for n in path:
                    n.visit_count += 1
                    n.value_sum -= VIRTUAL_LOSS
                if cur_root_idx >= 0:  # mirror the root child (path[1]) touched above
                    root_vc_arr[cur_root_idx] += 1.0
                    root_vsum_arr[cur_root_idx] -= VIRTUAL_LOSS

                batch_paths.append(path)
                batch_leaf_nodes.append(node)
                batch_games.append(sim_game)
                batch_root_idx.append(cur_root_idx)

                if sim_game.game_over:
                    batch_game_over.append(True)
                else:
                    batch_game_over.append(False)
                    obs = _build_obs_for_game(sim_game)
                    if use_server:
                        obs_np_buf[obs_count] = obs
                    else:
                        t = torch.from_numpy(obs)
                        if self._fp16:
                            t = t.half()
                        self._obs_buf[obs_count] = t
                    obs_count += 1

            # === BATCH EVALUATE ===
            # In feature-value mode the value head is unused; skip the value
            # transfer (~half of total .cpu() cost) and predict_value compute.
            skip_nn_value = self.feature_coefs is not None or \
                            self.value_head is not None
            val_np = None
            vh_np = None  # value head scalar V per leaf, when value_head set
            if obs_count > 0:
                if use_server:
                    pol_np, val_np = self.inference_client.evaluate_batch(
                        obs_np_buf, obs_count)
                    # No copy needed — shared memory is safe until next
                    # evaluate_batch call (worker is single-threaded)
                else:
                    with torch.inference_mode():
                        if self.value_head is not None:
                            # Run backbone once, apply BOTH policy and
                            # value heads from the shared features.
                            pol_logits, feats = \
                                self.net.forward_with_features(
                                    self._obs_buf[:obs_count])
                            vh_out = self.value_head(feats.float())
                            if self.value_head_type == 'spatial':
                                vh_scalars = vh_out.squeeze(-1)
                            elif self.value_head_target_type == 'density':
                                vh_scalars = (vh_out *
                                              self._horizon_weights).sum(dim=-1)
                            else:
                                vh_probs = torch.sigmoid(vh_out)
                                vh_scalars = (vh_probs *
                                              self._horizon_weights).sum(dim=-1)
                            vh_np = vh_scalars.float().cpu().numpy()
                            val_logits = None
                        else:
                            out = self.net(self._obs_buf[:obs_count])
                            # policy_only returns just pol_logits; dual-head
                            # returns (pol, val).
                            if isinstance(out, tuple):
                                pol_logits, val_logits = out
                            else:
                                pol_logits, val_logits = out, None
                            if not skip_nn_value and val_logits is not None:
                                values_t = self.net.predict_value(
                                    val_logits, max_val=self.max_score)
                    pol_np = pol_logits.float().cpu().numpy()  # fp32 for JIT
                    if not skip_nn_value and val_logits is not None:
                        val_np = values_t.cpu().numpy()

            # === VALUE NET BATCH EVAL (if separate value network) ===
            vnet_values = None
            if self.value_net is not None and obs_count > 0:
                vnet_device = next(self.value_net.parameters()).device
                vnet_dtype = next(self.value_net.parameters()).dtype
                if use_server:
                    vnet_obs = torch.from_numpy(
                        obs_np_buf[:obs_count].copy()).to(
                        device=vnet_device, dtype=vnet_dtype)
                else:
                    vnet_obs = self._obs_buf[:obs_count].to(
                        device=vnet_device, dtype=vnet_dtype)
                with torch.inference_mode():
                    vnet_logits = self.value_net(vnet_obs)
                    vnet_values = torch.sigmoid(
                        vnet_logits.squeeze(-1)).float().cpu().numpy()

            # === EXPAND + BACKUP ===
            nn_idx = 0
            for b in range(bs):
                path = batch_paths[b]

                if batch_game_over[b]:
                    if self.terminal_value is not None:
                        value = self.terminal_value
                    elif self.feature_coefs is not None:
                        n_next = self._fill_next_ball_buffers(batch_games[b])
                        value = float(_evaluate_features_linear(
                            batch_games[b].board,
                            self._nb_r, self._nb_c, self._nb_col, n_next,
                            self.feature_coefs, self.feature_means,
                            self.feature_stds, self.feature_bias))
                    elif self.value_head is not None:
                        # Terminal boards weren't in the batched forward;
                        # evaluate alone. Cheap (1 forward) and rare.
                        value = self._value_head_eval_single(batch_games[b])
                    elif self.value_head_via_server:
                        # Survival head: terminal board ⇒ no future, V=0.
                        value = 0.0
                    elif self.value_net is not None:
                        value = 0.0
                    else:
                        value = float(batch_games[b].score)
                elif self.heuristic_value:
                    value = _evaluate_board(batch_games[b].board)
                else:
                    # Expand with policy priors
                    node = batch_leaf_nodes[b]
                    k, flat_idx, pri = _legal_priors_jit(
                        batch_games[b].board, pol_np[nn_idx], top_k)
                    ch = node.children
                    for i in range(k):
                        action_key = int(flat_idx[i])
                        if action_key not in ch:
                            ch[action_key] = Node(prior=float(pri[i]))
                    # Value source priority: features > value_head > vnet > NN val head
                    if self.feature_coefs is not None:
                        n_next = self._fill_next_ball_buffers(batch_games[b])
                        value = float(_evaluate_features_linear(
                            batch_games[b].board,
                            self._nb_r, self._nb_c, self._nb_col, n_next,
                            self.feature_coefs, self.feature_means,
                            self.feature_stds, self.feature_bias))
                    elif vh_np is not None:
                        value = float(vh_np[nn_idx])
                    elif vnet_values is not None:
                        value = float(vnet_values[nn_idx])
                    else:
                        value = float(val_np[nn_idx])
                    nn_idx += 1

                min_q = min(min_q, value)
                max_q = max(max_q, value)

                for n in path:
                    n.value_sum += VIRTUAL_LOSS + value
                ridx = batch_root_idx[b]
                if ridx >= 0:  # mirror the root child's value_sum update
                    root_vsum_arr[ridx] += VIRTUAL_LOSS + value

            sims_done += bs

            # Exact greedy-action early stop. If the most-visited root child
            # cannot be overtaken even if every remaining sim went to the
            # runner-up, the final argmax is fixed. Eval-only — self-play
            # policy targets need the full visit distribution.
            if self.early_stop and not return_policy and temperature == 0:
                remaining = num_sims - sims_done
                if remaining > 0 and len(root.children) > 1:
                    top_1 = 0
                    top_2 = 0
                    for child in root.children.values():
                        vc = child.visit_count
                        if vc >= top_1:
                            top_2 = top_1
                            top_1 = vc
                        elif vc > top_2:
                            top_2 = vc
                    if top_1 > top_2 + remaining:
                        break

        # Build policy target from visit counts
        if return_policy or temperature > 0:
            actions = list(root.children.keys())
            visits = np.array([root.children[a].visit_count
                               for a in actions], dtype=np.float32)

        if return_policy:
            # Build full 6561-dim policy target
            # Actions are already flat indices — write directly
            policy_target = np.zeros(NUM_MOVES, dtype=np.float32)
            visit_sum = visits.sum()
            if visit_sum > 0:
                for i, a in enumerate(actions):
                    policy_target[a] = visits[i] / visit_sum

        # Select move
        if temperature > 0 and len(root.children) > 0:
            # Temperature-weighted sampling
            adjusted = visits ** (1.0 / temperature)
            probs = adjusted / adjusted.sum()
            # Weighted sampling using our deterministic RNG
            u = self._sim_rng.next_f64()
            cumsum = 0.0
            chosen_idx = len(actions) - 1  # fallback to last
            for ci in range(len(actions)):
                cumsum += probs[ci]
                if u < cumsum:
                    chosen_idx = ci
                    break
            flat_action = actions[chosen_idx]
        else:
            # Greedy argmax
            flat_action = max(root.children.items(),
                              key=lambda x: x[1].visit_count)[0]

        # Override threshold: only override policy's top move if MCTS
        # has significantly more visits on a different move.
        # Prevents destructive coin-flip overrides on near-ties.
        if self.override_threshold > 0 and temperature == 0:
            # Find policy's preferred move (highest prior)
            policy_action = max(root.children.items(),
                                key=lambda x: x[1].prior)[0]
            mcts_action = max(root.children.items(),
                              key=lambda x: x[1].visit_count)[0]
            if mcts_action != policy_action:
                pol_visits = root.children[policy_action].visit_count
                mcts_visits = root.children[mcts_action].visit_count
                if mcts_visits <= pol_visits * (1 + self.override_threshold):
                    flat_action = policy_action  # trust policy on near-ties

        # Decode flat action to tuple format for callers
        action = _flat_to_action(flat_action)

        self._last_root = root
        self._last_min_q = min_q
        self._last_max_q = max_q

        if return_policy:
            return action, policy_target
        return action

    def last_root_record(self, top_k=20):
        """Rich per-root training record from the most recent search(), shared by
        self-play and crisis recording so the corpus schema is uniform.

        Returns a dict of top-K candidates (ranked by visit count) with, per move:
        flat move idx, visit count, CLEAN pre-Dirichlet prior logit-proxy (log prob),
        and root Q (value_sum/visit_count); plus root_value and the search's q_min/q_max
        (for advantage/Gumbel normalization). Supports visit / advantage-softmax / Gumbel
        targets and disagreement weighting without re-search.
        """
        root = self._last_root
        if root is None or not root.children:
            return None
        raw = getattr(self, '_last_raw_priors', None) or {}
        items = sorted(root.children.items(),
                       key=lambda kv: kv[1].visit_count, reverse=True)[:top_k]
        moves, visits, priors, qs = [], [], [], []
        for action, node in items:
            moves.append(int(action))
            visits.append(int(node.visit_count))
            p = raw.get(action, node.prior)
            priors.append(float(np.log(p + 1e-8)))           # clean prior as log-prob
            qs.append(float(node.value_sum / node.visit_count)
                      if node.visit_count > 0 else float(self._last_root_value))
        return {
            'cand_moves': moves, 'cand_visits': visits,
            'cand_prior': priors, 'cand_q': qs,
            'root_value': float(getattr(self, '_last_root_value', 0.0)),
            'q_min': float(self._last_min_q), 'q_max': float(self._last_max_q),
        }


def make_mcts_player(net, device, max_score=30000.0,
                     num_simulations=400, c_puct=2.5, top_k=30,
                     batch_size=16, value_net=None, terminal_value=None,
                     override_threshold=0.0, feature_weights_path=None,
                     early_stop=False, q_weight=1.0,
                     value_head_path=None):
    """Create MCTS player function for use with evaluate."""
    mcts = MCTS(net, device, max_score=max_score,
                num_simulations=num_simulations,
                c_puct=c_puct, top_k=top_k, batch_size=batch_size,
                value_net=value_net, terminal_value=terminal_value,
                override_threshold=override_threshold,
                feature_weights_path=feature_weights_path,
                early_stop=early_stop, q_weight=q_weight,
                value_head_path=value_head_path)

    def player(game):
        return mcts.search(game)

    return player
