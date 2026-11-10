"""Task 4 — Reference scaled dot-product attention and tiled online-softmax attention.

Shapes: q [B, H, Lq, D], k/v [B, H, Lk, D]. Causal masking uses the "bottom-right"
alignment used by decoders with a KV cache: query i (0..Lq-1) sits at absolute position
(Lk - Lq + i) and may attend to keys j <= Lk - Lq + i. For Lq == Lk this is the usual
lower-triangular mask.

Online softmax (Milakov & Gimelshein 2018; FlashAttention, Dao et al. 2022): process K/V
in blocks, keep per-row running max m, normaliser l and un-normalised accumulator acc:
    m_new = max(m, rowmax(S_blk))
    p     = exp(S_blk - m_new)
    alpha = exp(m - m_new)
    l     = alpha * l + rowsum(p)
    acc   = alpha * acc + p @ V_blk
Final output = acc / l. The full Lq x Lk score matrix is never materialised: peak extra
memory is O(Bq x Bk) per (batch, head) instead of O(Lq x Lk).
"""
from __future__ import annotations

import math

import torch


def causal_mask(lq: int, lk: int, device=None) -> torch.Tensor:
    """Boolean mask, True = allowed. Bottom-right aligned."""
    i = torch.arange(lq, device=device).unsqueeze(1) + (lk - lq)
    j = torch.arange(lk, device=device).unsqueeze(0)
    return j <= i


def reference_attention(q, k, v, causal: bool = True, scale: float | None = None):
    """Naive SDPA: softmax(QK^T / sqrt(d) + mask) V with max-subtraction for stability.
    Computed in float32 for use as the numerical reference."""
    in_dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-1, -2)) * scale            # [B,H,Lq,Lk]
    if causal:
        s = s.masked_fill(~causal_mask(q.shape[-2], k.shape[-2], q.device), float("-inf"))
    m = s.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))  # fully-masked rows
    p = torch.exp(s - m)
    l = p.sum(dim=-1, keepdim=True)
    out = torch.matmul(p, v) / l.clamp_min(1e-30)
    return out.to(in_dtype)


def tiled_attention(q, k, v, causal: bool = True, block_q: int = 64, block_k: int = 64,
                    scale: float | None = None):
    """Blocked attention with online softmax (pure PyTorch, educational).
    Accumulates in float32 regardless of input dtype."""
    B, H, Lq, D = q.shape
    Lk = k.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    offset = Lk - Lq
    out = torch.empty((B, H, Lq, v.shape[-1]), dtype=q.dtype, device=q.device)
    for qs in range(0, Lq, block_q):
        qe = min(qs + block_q, Lq)
        qb = q[:, :, qs:qe].float() * scale
        nq = qe - qs
        m = torch.full((B, H, nq, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, nq, 1), device=q.device)
        acc = torch.zeros((B, H, nq, v.shape[-1]), device=q.device)
        # last key any query in this block may see (causal): skip fully-masked K blocks
        k_limit = min(Lk, qe - 1 + offset + 1) if causal else Lk
        for ks in range(0, k_limit, block_k):
            ke = min(ks + block_k, Lk)
            s = torch.matmul(qb, k[:, :, ks:ke].float().transpose(-1, -2))   # [B,H,nq,nk]
            if causal and ke - 1 > qs + offset:  # block crosses the diagonal -> mask
                qi = torch.arange(qs, qe, device=q.device).unsqueeze(1) + offset
                kj = torch.arange(ks, ke, device=q.device).unsqueeze(0)
                s = s.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            p = torch.exp(s - m_safe)
            alpha = torch.exp(torch.where(torch.isfinite(m), m - m_safe, torch.full_like(m, float("-inf"))))
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(p, v[:, :, ks:ke].float())
            m = m_new
        out[:, :, qs:qe] = (acc / l.clamp_min(1e-30)).to(q.dtype)
    return out
