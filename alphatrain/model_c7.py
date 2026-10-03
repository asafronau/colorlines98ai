"""Color-equivariant (S7) policy: one feature slot per color with shared weights (HISTORY 279).

Renaming the 7 colors changes nothing in the game: what matters is which balls share a color and how many of each
there are, not whether they are yellow or blue. A plain CNN reads each color through its own input plane, so
"four yellow in a row" and "four blue in a row" go through different weights and are only made to agree by color
augmentation (the best actor still changes its move on ~7% of states when the colors are renamed). This net carries
one feature slot per color, processed with SHARED weights (DeepSets-style): every layer maps each slot through the
same filters, adds the mean over slots and a color-free shared stream, and updates the shared stream from the slot
mean -- the general linear map that commutes with renaming the colors. A pattern is learned once for all colors,
and renaming the colors only permutes the slots.

Layout: the trunk is ONE (B, 7*K + S, 9, 9) tensor, [slot 0 (K) | ... | slot 6 (K) | shared (S)], and each
equivariant layer is ONE dense convolution whose weight is assembled from the shared blocks (dense_slot_weight). In
the trunk this is exactly the DeepSets map; on a GPU it runs as a plain ResNet of width 7K + S (a version with
separate per-slot convolutions, means and broadcasts ran 4-10x slower on MPS than its FLOPs). BatchNorm statistics
are shared across the 7 slots. freeze() turns the net into plain Conv2d/BatchNorm2d modules for export and folding.

The head reads the MOVING ball's own color slot, at the source and at every destination ("what does color c look
like around d?"), so the logits are exactly invariant to any color renaming. The policy is PAIR2-shaped
(bilinear source/destination embeddings + biases + line/adjacency-gated terms, alphatrain/model.py), with the
destination embedding taken from the source ball's color slot.

Input: the standard 18-channel observation (no format change; the C++ engine is unchanged). The slot planes are
decoded inside the net: board==c (channels 0-6), preview ball of color c (channels 8-10 hold color/7 at the preview
cells) and the share of the board's cells holding color c (a constant plane). Shared planes: empty, preview mask,
component area, line potentials H/V/D1/D2 and max line length (channels 7 and 11-17, all color-free).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval as fuse_conv_bn

NCOL = 7
SHARED_IN = (7, 11, 12, 13, 14, 15, 16, 17)
SLOT_IN = 3


def slot_mean(h):
    """Mean over the color slots of (B, 7, K, 9, 9) -> (B, K, 9, 9), reduced on a 3-D view (MPS reduces a 5-D middle
    dim ~20x slower: 4.1 vs 0.2 us per position at K=24)."""
    b, n, k, hh, ww = h.shape
    return h.reshape(b, n, k * hh * ww).mean(1).reshape(b, k, hh, ww)


def own_slot(h, board):
    """Each cell's own color slot: sum_c h[:, c] * board[:, c] -> (B, K, 9, 9) (zero on empty cells), on 4-D views."""
    b, n, k, hh, ww = h.shape
    return (h.reshape(b, n, k, hh * ww) * board.reshape(b, n, 1, hh * ww)).sum(1).reshape(b, k, hh, ww)


def split_observation(x):
    """(B, 18, 9, 9) observation -> slot planes (B, 7, 3, 9, 9), shared planes (B, 8, 9, 9), board one-hot (B, 7, 9, 9)."""
    board = x[:, :NCOL]
    code = torch.round(x[:, 8:11].sum(1, keepdim=True) * 7)                 # preview color 1..7 at its cell, else 0
    cols = torch.arange(1, NCOL + 1, device=x.device, dtype=x.dtype).view(1, NCOL, 1, 1)
    preview = (code == cols).to(x.dtype)
    share = board.sum((2, 3), keepdim=True).expand_as(board) / 81.0
    slots = torch.stack([board, preview, share], 2)
    shared = torch.cat([x[:, 7:8], x[:, 11:18]], 1)
    return slots, shared, board


def dense_slot_weight(a, b, c, e=None, d=None):
    """Weight of the dense convolution equal to the color-equivariant map [7 slots x Ki | Si] -> [7 slots x Ko | So]:
    slot c <- slot c': a [c = c'] + b / 7;  slot <- shared: c;  shared <- slot: e / 7;  shared <- shared: d.
    Without e, d: no shared output (slots only)."""
    n = NCOL
    ko, ki, kh, kw = a.shape
    eye = torch.eye(n, dtype=a.dtype, device=a.device).view(n, 1, n, 1, 1, 1)
    w_ss = (eye * a.view(1, ko, 1, ki, kh, kw) + b.view(1, ko, 1, ki, kh, kw) / n).reshape(n * ko, n * ki, kh, kw)
    w_sz = c.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * ko, c.shape[1], kh, kw)
    top = torch.cat([w_ss, w_sz], 1)
    if e is None:
        return top
    w_zs = (e / n).unsqueeze(1).expand(-1, n, -1, -1, -1).reshape(e.shape[0], n * ki, kh, kw)
    return torch.cat([top, torch.cat([w_zs, d], 1)], 0)


class SlotBN(nn.Module):
    """BatchNorm over the 7 x K slot channels with statistics shared across the colors."""
    def __init__(self, k):
        super().__init__()
        self.bn = nn.BatchNorm2d(k)

    def forward(self, x):
        b, nk, hh, ww = x.shape
        k = nk // NCOL
        y = self.bn(x.reshape(b, NCOL, k, hh * ww).transpose(1, 2))        # (B, K, 7, HW): K channels
        return y.transpose(1, 2).reshape(b, nk, hh, ww)


def tied_bn(x, bnh, bnz, k):
    """[7 slots x K | S] tensor: SlotBN on the slots, plain BatchNorm on the shared channels."""
    nk = NCOL * k
    return torch.cat([bnh(x[:, :nk]), bnz(x[:, nk:])], 1)


def merged_bn(bnh, bnz=None):
    """One BatchNorm2d equal (in eval mode) to SlotBN bnh (+ BatchNorm bnz) on [7 slots x K (| S)] channels."""
    hb = bnh.bn
    parts = [(hb, NCOL)] + ([(bnz, 1)] if bnz is not None else [])
    out = nn.BatchNorm2d(sum(m.num_features * r for m, r in parts), eps=hb.eps)
    with torch.no_grad():
        for name in ('weight', 'bias', 'running_mean', 'running_var'):
            getattr(out, name).copy_(torch.cat([getattr(m, name).repeat(r) for m, r in parts]))
    return out.to(device=hb.weight.device, dtype=hb.weight.dtype).train(False)


class SlotMix(nn.Module):
    """Color-equivariant convolution [7 slots x k_in | s_in] -> [7 slots x k_out | s_out]: h'_c = A*h_c + B*mean(h) +
    C*z, z' = D*z + E*mean(h). Free weights: a = A, g = [[B, C], [E, D]] (g reads [mean(h), z], writes [slot, z]);
    run as one dense convolution."""
    def __init__(self, k_in, s_in, k_out, s_out, ksize=3):
        super().__init__()
        self.k_in, self.k_out, self.ksize = k_in, k_out, ksize
        self.a = nn.Conv2d(k_in, k_out, ksize, padding=ksize // 2, bias=False)
        self.g = nn.Conv2d(k_in + s_in, k_out + s_out, ksize, padding=ksize // 2, bias=False)

    def dense(self):
        ki, ko, g = self.k_in, self.k_out, self.g.weight
        w = dense_slot_weight(self.a.weight, g[:ko, :ki], g[:ko, ki:], g[ko:, :ki], g[ko:, ki:])
        bias = None if self.g.bias is None else torch.cat([self.g.bias[:ko].repeat(NCOL), self.g.bias[ko:]])
        return w, bias

    def forward(self, x):
        w, bias = self.dense()
        return F.conv2d(x, w, bias, padding=self.ksize // 2)

    def frozen(self):
        w, bias = self.dense()
        conv = nn.Conv2d(w.shape[1], w.shape[0], self.ksize, padding=self.ksize // 2, bias=bias is not None)
        with torch.no_grad():
            conv.weight.copy_(w)
            if bias is not None:
                conv.bias.copy_(bias)
        return conv.to(device=w.device, dtype=w.dtype).train(False)


class SlotResBlock(nn.Module):
    """Pre-activation residual block (as alphatrain.model.ResBlock) on the [7 slots x K | S] tensor."""
    def __init__(self, k, s):
        super().__init__()
        self.k = k
        self.bn1h, self.bn1z, self.mix1 = SlotBN(k), nn.BatchNorm2d(s), SlotMix(k, s, k, s)
        self.bn2h, self.bn2z, self.mix2 = SlotBN(k), nn.BatchNorm2d(s), SlotMix(k, s, k, s)
        self.bn1 = None     # single BatchNorm2d each after freeze()
        self.bn2 = None

    def forward(self, x):
        y = self.bn1(x) if self.bn1 is not None else tied_bn(x, self.bn1h, self.bn1z, self.k)
        y = self.mix1(F.relu(y))
        y = self.bn2(y) if self.bn2 is not None else tied_bn(y, self.bn2h, self.bn2z, self.k)
        return x + self.mix2(F.relu(y))


class PolicyNetC7(nn.Module):
    """Color-equivariant trunk (slot width K, shared width S) + color-selecting PAIR2 head. logits (B, 6561),
    index = source * 81 + destination, exactly invariant to renaming the colors."""

    def __init__(self, num_blocks=18, slot_channels=24, shared_channels=48, policy_channels=64, pair_dim=64):
        super().__init__()
        k, s, p = slot_channels, shared_channels, policy_channels
        self.num_blocks, self.slot_channels, self.shared_channels = num_blocks, k, s
        self.policy_channels, self.pair_dim = p, pair_dim
        self.slot_stem = SlotMix(SLOT_IN, len(SHARED_IN), k, s)
        self.stem_bnh, self.stem_bnz = SlotBN(k), nn.BatchNorm2d(s)
        self.blocks = nn.Sequential(*[SlotResBlock(k, s) for _ in range(num_blocks)])
        self.trunk_bnh, self.trunk_bnz = SlotBN(k), nn.BatchNorm2d(s)
        # Head features: source = [own-color slot, z, mean slot]; destination for color c = [slot c, z, mean slot].
        self.src_conv = nn.Conv2d(2 * k + s, p, 1, bias=False)
        self.src_bn = nn.BatchNorm2d(p)
        self.dst_conv_slot = nn.Conv2d(k, p, 1, bias=False)      # slot c <- slot c
        self.dst_conv_rest = nn.Conv2d(k + s, p, 1, bias=False)  # slot c <- [z, mean slot]
        self.dst_bn = SlotBN(p)
        for name in ('pair', 'line', 'adj'):
            setattr(self, f'{name}_src', nn.Conv2d(p, pair_dim, 1))
            setattr(self, f'{name}_dst', nn.Conv2d(p, pair_dim, 1))
        self.pair_sbias = nn.Conv2d(p, 1, 1)
        self.pair_dbias = nn.Conv2d(p, 1, 1)
        line = torch.zeros(81, 81); adj = torch.zeros(81, 81)
        for s_ in range(81):
            for d_ in range(81):
                dr, dc = s_ // 9 - d_ // 9, s_ % 9 - d_ % 9
                if 1 <= max(abs(dr), abs(dc)) <= 4 and (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
                    line[s_, d_] = 1.0
                if abs(dr) + abs(dc) == 1:
                    adj[s_, d_] = 1.0
        self.register_buffer('line_gate', line, persistent=False)
        self.register_buffer('adj_gate', adj, persistent=False)
        # Set by freeze(): merged BatchNorms and the dense destination convolution.
        self.stem_bn = self.trunk_bn = self.dst_conv = self.dst_bn_merged = None

    def trunk(self, x):
        """-> dense trunk features (B, 7K + S, 9, 9) after the final BatchNorm + ReLU, and the board one-hot."""
        slots, shared, board = split_observation(x)
        b, k = x.shape[0], self.slot_channels
        t = self.slot_stem(torch.cat([slots.reshape(b, NCOL * SLOT_IN, 9, 9), shared], 1))
        t = F.relu(self.stem_bn(t) if self.stem_bn is not None else tied_bn(t, self.stem_bnh, self.stem_bnz, k))
        t = self.blocks(t)
        t = F.relu(self.trunk_bn(t) if self.trunk_bn is not None else tied_bn(t, self.trunk_bnh, self.trunk_bnz, k))
        return t, board

    def _slots(self, t):
        b, nk = t.shape[0], NCOL * self.slot_channels
        return t[:, :nk].reshape(b, NCOL, self.slot_channels, 9, 9), t[:, nk:]

    def _invariant_features(self, t, board):
        h, z = self._slots(t)
        return torch.cat([own_slot(h, board), z, slot_mean(h)], 1)   # own color slot (0 on empty cells), shared, mean

    def backbone_features(self, x):
        """Color-invariant per-cell features (B, 2K+S, 9, 9) for frozen-backbone value heads."""
        return self._invariant_features(*self.trunk(x))

    def _dst_dense(self):
        s = self.shared_channels
        rest = self.dst_conv_rest.weight                              # input channels [z (S), mean slot (K)]
        w = dense_slot_weight(self.dst_conv_slot.weight, rest[:, s:], rest[:, :s])
        bias = None if self.dst_conv_rest.bias is None else self.dst_conv_rest.bias.repeat(NCOL)
        return w, bias

    def _policy(self, t, board):
        b, n, pdim = t.shape[0], NCOL, self.pair_dim
        ps = F.relu(self.src_bn(self.src_conv(self._invariant_features(t, board))))             # (B, P, 9, 9)
        if self.dst_conv is not None:
            pd = F.relu(self.dst_bn_merged(self.dst_conv(t)))
        else:
            w, bias = self._dst_dense()
            pd = F.relu(self.dst_bn(F.conv2d(t, w, bias)))                                     # (B, 7P, 9, 9)
        pd = pd.reshape(b * n, -1, 9, 9)                                                        # (B*7, P, 9, 9)
        # Source mask per stacked color block: [color(s) = c] repeated over the pair_dim rows of block c.
        mask = board.reshape(b, n, 1, 81).repeat(1, 1, pdim, 1).reshape(b, n * pdim, 81)
        sc = pdim ** -0.5

        def bilinear(src, dst):
            # sum_c [color(s) = c] u(s) . w_c(d): one bmm over the 7 stacked color blocks.
            u = src(ps).reshape(b, pdim, 81).repeat(1, n, 1) * mask
            w = dst(pd).reshape(b, n * pdim, 81)
            return torch.bmm(u.transpose(1, 2), w) * sc                                         # (B, s, d)

        logits = bilinear(self.pair_src, self.pair_dst)
        logits = logits + self.pair_sbias(ps).reshape(b, 81, 1)
        logits = logits + torch.bmm(board.reshape(b, n, 81).transpose(1, 2), self.pair_dbias(pd).reshape(b, n, 81))
        logits = logits + bilinear(self.line_src, self.line_dst) * self.line_gate
        logits = logits + bilinear(self.adj_src, self.adj_dst) * self.adj_gate
        return logits.reshape(b, 81 * 81)

    def forward(self, x):
        return self._policy(*self.trunk(x))

    def forward_with_features(self, x):
        t, board = self.trunk(x)
        return self._policy(t, board), self._invariant_features(t, board)

    @torch.no_grad()
    def freeze(self):
        """Inference/export form (eval mode): every equivariant convolution becomes a plain Conv2d with its dense
        weight and every slot-tied BatchNorm pair one BatchNorm2d with repeated statistics. Same function; the trunk
        is then an ordinary pre-activation ResNet of width 7K + S. No longer trainable as a slot net."""
        if self.training:
            raise RuntimeError('freeze() needs eval mode (BatchNorm running statistics)')
        if self.stem_bn is not None:
            return self
        self.slot_stem = self.slot_stem.frozen()
        self.stem_bn = merged_bn(self.stem_bnh, self.stem_bnz)
        for blk in self.blocks:
            blk.mix1, blk.mix2 = blk.mix1.frozen(), blk.mix2.frozen()
            blk.bn1, blk.bn2 = merged_bn(blk.bn1h, blk.bn1z), merged_bn(blk.bn2h, blk.bn2z)
        self.trunk_bn = merged_bn(self.trunk_bnh, self.trunk_bnz)
        w, bias = self._dst_dense()
        conv = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=bias is not None)
        conv.weight.copy_(w)
        if bias is not None:
            conv.bias.copy_(bias)
        self.dst_conv = conv.to(device=w.device, dtype=w.dtype).train(False)
        self.dst_bn_merged = merged_bn(self.dst_bn)
        return self

    @torch.no_grad()
    def fold_batchnorm(self):
        """Inference/export: freeze(), then fold every BatchNorm that directly follows a convolution (the stem, each
        block's second, both head BNs) into it, as export_ts.fold_batchnorm does for PolicyNet. Pre-activation BNs
        and the trunk BN follow a residual sum and cannot fold. Returns the number of BatchNorms folded."""
        self.freeze()
        self.slot_stem = fuse_conv_bn(self.slot_stem, self.stem_bn)
        self.stem_bn = nn.Identity()
        for blk in self.blocks:
            blk.mix1 = fuse_conv_bn(blk.mix1, blk.bn2)
            blk.bn2 = nn.Identity()
        self.src_conv = fuse_conv_bn(self.src_conv, self.src_bn)
        self.src_bn = nn.Identity()
        self.dst_conv = fuse_conv_bn(self.dst_conv, self.dst_bn_merged)
        self.dst_bn_merged = nn.Identity()
        return 1 + len(self.blocks) + 2


def is_c7_state(state):
    return any(k.endswith('slot_stem.a.weight') for k in state)


def c7_kwargs_from_state(state):
    a = next(v for k, v in state.items() if k.endswith('slot_stem.a.weight'))
    g = next(v for k, v in state.items() if k.endswith('slot_stem.g.weight'))
    src = next(v for k, v in state.items() if k.endswith('pair_src.weight'))
    nb = sum(1 for k in state if k.endswith('.mix1.a.weight'))
    k = int(a.shape[0])
    return dict(num_blocks=nb, slot_channels=k, shared_channels=int(g.shape[0]) - k,
                policy_channels=int(src.shape[1]), pair_dim=int(src.shape[0]))
