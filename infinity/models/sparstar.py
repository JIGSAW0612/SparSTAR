# Copyright (c) 2026 SparSTAR Authors
# SPDX-License-Identifier: MIT
"""
[SparSTAR] Training-free block-sparse attention for InfinityStar inference.

At every expensive late scale (and for every attention head), SparSTAR
  1. scores contiguous key blocks from the *current* query and key activations
     (dot product between block-averaged queries and block-averaged keys),
  2. always keeps the conditioning context, i.e. the KV-cached prefix (reference scales + text),
  3. keeps the top-scoring key blocks of the current scale up to a per-scale density, and
  4. executes only the selected blocks with FlexAttention on a forward-only path.
The selection is recomputed at every target scale; no mask is reused across scales or clips.
Every token and every refinement scale is still generated, and all other scales run dense attention.

Entry points:
  configure(args)          read the SparSTAR options from `args` (see infinity/utils/arg_util.py)
  is_sparse_scale(si)      whether scale `si` runs block-sparse attention
  block_sparse_attention   the sparse attention itself, called from SelfAttention.forward (infinity/models/basic.py)
  parse_scale_list         helper for the comma-separated scale lists used by the generation loop
"""
import math
from typing import Dict, Set

import torch
import torch.nn.functional as F

# Paper setting for 720p 5 s generation (Table 1): the last four scales of the video clip.
DEFAULT_DENSITY = '26:0.625,27:0.1875,28:0.1875,29:0.09375'

_CONFIG = {'enabled': False, 'density': {}, 'block_size': 128}
_FLEX = None


def parse_density(spec: str) -> Dict[int, float]:
    """'26:0.625,27:0.1875' -> {26: 0.625, 27: 0.1875} (scale index -> fraction of key blocks kept)."""
    density = {}
    for item in spec.split(','):
        if item.strip():
            scale_ind, ratio = item.split(':')
            density[int(scale_ind)] = float(ratio)
    return density


def parse_scale_list(spec: str) -> Set[int]:
    """'28,29' -> {28, 29}"""
    return {int(x) for x in str(spec).split(',') if x.strip()}


def configure(args) -> None:
    """Read the SparSTAR options from `args`. Called at the start of every generation."""
    _CONFIG['enabled'] = bool(getattr(args, 'sparstar', 0))
    _CONFIG['density'] = parse_density(getattr(args, 'sparstar_density', DEFAULT_DENSITY)) if _CONFIG['enabled'] else {}
    _CONFIG['block_size'] = int(getattr(args, 'sparstar_block_size', 128))


def reset() -> None:
    _CONFIG['enabled'] = False
    _CONFIG['density'] = {}


def is_sparse_scale(scale_ind) -> bool:
    # scale_ind is 't0' for the text-prefix forward and an int for visual scales
    return _CONFIG['enabled'] and isinstance(scale_ind, int) and scale_ind in _CONFIG['density']


def _flex_attention():
    # FlexAttention must be compiled: the eager version materializes the full score matrix.
    # One kernel is compiled per (scale, batch) shape, so the recompile limits are raised to
    # keep dynamo from falling back to the eager path.
    global _FLEX
    if _FLEX is None:
        import torch._dynamo
        from torch.nn.attention.flex_attention import flex_attention
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.accumulated_cache_size_limit = 512
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def _keep_all(b, h, q_idx, kv_idx):
    # The sparsity lives entirely in the block list; selected blocks are attended in full.
    return q_idx >= 0


def _block_mean(x: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """(B, H, L, D) -> (B, H, num_blocks, D): average of each contiguous block along the sequence."""
    B, H, L, D = x.shape
    return F.adaptive_avg_pool1d(x.transpose(2, 3).reshape(B * H, D, L), num_blocks).reshape(B, H, D, num_blocks).transpose(2, 3)


@torch.no_grad()
def block_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, scale_ind: int) -> torch.Tensor:
    """
    q: (B, H, Lq, D) queries of the current scale.
    k, v: (B, H, Lk, D) keys / values laid out as [conditioning prefix (Lk - Lq) | current scale (Lq)].
    Returns the attention output, (B, H, Lq, D).
    """
    from torch.nn.attention.flex_attention import BlockMask  # torch==2.5.1 signature
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    blk = _CONFIG['block_size']
    density = _CONFIG['density'][scale_ind]
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    num_q_blocks = math.ceil(Lq / blk)
    num_kv_blocks = math.ceil(Lk / blk)
    num_prefix_blocks = math.ceil((Lk - Lq) / blk)
    num_own_blocks = num_kv_blocks - num_prefix_blocks

    # (1) Block scores from the current query/key activations, per head and query block.
    score = torch.einsum('bhqd,bhkd->bhqk', _block_mean(q, num_q_blocks), _block_mean(k, num_kv_blocks)) * softmax_scale

    # (2) The conditioning prefix is always kept (it is ranked ahead of every current-scale block).
    score[..., :num_prefix_blocks] = 1e9

    # (3) Keep the prefix plus the top ceil(density * #current-scale blocks) blocks.
    num_keep = min(num_kv_blocks, num_prefix_blocks + max(1, math.ceil(density * num_own_blocks)))
    kv_indices = score.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
    kv_num_blocks = torch.full((B, H, num_q_blocks), num_keep, dtype=torch.int32, device=q.device)

    # (4) Forward-only block mask. The q-side tables (q_num_blocks / q_indices) are only read by the
    #     backward pass, so placeholders are passed instead of building them (BlockMask.from_kv_blocks
    #     would transpose the whole kv table on every call).
    q_num_blocks = torch.zeros((B, H, num_kv_blocks), dtype=torch.int32, device=q.device)
    q_indices = torch.zeros((B, H, num_kv_blocks, 1), dtype=torch.int32, device=q.device)
    block_mask = BlockMask(
        kv_num_blocks, kv_indices, None, None,
        q_num_blocks, q_indices, None, None,
        (blk, blk), _keep_all,
    )
    return _flex_attention()(q, k, v, block_mask=block_mask, scale=softmax_scale)
