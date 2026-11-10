"""Task 3 — Explicit KV-cached greedy decoding vs. a no-cache reference.

We do NOT call model.generate(). Both decoders drive the Transformer's forward pass
directly:

* ``greedy_no_cache``: at every step the FULL sequence (prompt + generated tokens) is
  re-encoded with use_cache=False. Cost per step O(t) tokens -> O(T^2) total.
* ``greedy_kv_cache``: one *prefill* pass over the prompt fills a DynamicCache with the
  per-layer key/value tensors [batch, n_kv_heads, seq, head_dim]. Each *decode* step feeds
  only the newest token, with
    - ``position_ids = cache_position = [past_len]`` so RoPE rotates the new query/key at
      its true absolute position (feeding position 0 would silently corrupt attention);
    - an attention mask of length past_len+1 (all ones: a single query may attend to all
      cached keys, which is exactly the causal constraint for the last position);
  the model appends the new K/V to the cache (cache growth: +1 position per layer per step,
  memory = 2 * n_layers * n_kv_heads * head_dim * seq * bytes).

Under fp16 the two paths use different matmul shapes (full vs. 1-row), so logits can differ
in the last bits; a greedy argmax flip is possible only when the top-2 logit margin is tiny.
``compare_paths`` reports the first divergence and its margin when this happens.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
from transformers import DynamicCache


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def greedy_no_cache(model, prompt_ids: List[int], max_new_tokens: int, eos_ids=(),
                    stop_on_eos: bool = True) -> Dict:
    dev = model.device
    seq = torch.tensor([prompt_ids], device=dev)
    out: List[int] = []
    margins: List[float] = []
    _sync(dev)
    t0 = time.perf_counter()
    t_first = None
    for step in range(max_new_tokens):
        logits = model(input_ids=seq, attention_mask=torch.ones_like(seq),
                       use_cache=False).logits[0, -1].float()
        top2 = torch.topk(logits, 2)
        nxt = int(top2.indices[0])
        margins.append(float(top2.values[0] - top2.values[1]))
        if step == 0:
            _sync(dev)
            t_first = time.perf_counter() - t0
        out.append(nxt)
        if stop_on_eos and nxt in eos_ids:
            break
        seq = torch.cat([seq, torch.tensor([[nxt]], device=dev)], dim=1)
    _sync(dev)
    total = time.perf_counter() - t0
    return {"tokens": out, "margins": margins, "prefill_s": t_first,
            "decode_s": total - t_first, "total_s": total}


@torch.no_grad()
def prefill(model, prompt_ids: List[int], cache: Optional[DynamicCache] = None):
    """Run the prompt through the model once, filling the KV cache."""
    dev = model.device
    cache = cache if cache is not None else DynamicCache()
    ids = torch.tensor([prompt_ids], device=dev)
    n = ids.shape[1]
    pos = torch.arange(n, device=dev)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                position_ids=pos.unsqueeze(0), cache_position=pos,
                past_key_values=cache, use_cache=True)
    return out.logits[0, -1].float(), out.past_key_values


@torch.no_grad()
def decode_step(model, token: int, cache: DynamicCache):
    """Feed ONE token; its position is the current cache length."""
    dev = model.device
    past = cache.get_seq_length()
    ids = torch.tensor([[token]], device=dev)
    pos = torch.tensor([past], device=dev)
    mask = torch.ones((1, past + 1), dtype=torch.long, device=dev)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos.unsqueeze(0),
                cache_position=pos, past_key_values=cache, use_cache=True)
    return out.logits[0, -1].float(), out.past_key_values


@torch.no_grad()
def greedy_kv_cache(model, prompt_ids: List[int], max_new_tokens: int, eos_ids=(),
                    stop_on_eos: bool = True) -> Dict:
    dev = model.device
    _sync(dev)
    t0 = time.perf_counter()
    logits, cache = prefill(model, prompt_ids)
    _sync(dev)
    t_prefill = time.perf_counter() - t0
    out: List[int] = []
    margins: List[float] = []
    cache_lens = [cache.get_seq_length()]
    for step in range(max_new_tokens):
        top2 = torch.topk(logits, 2)
        nxt = int(top2.indices[0])
        margins.append(float(top2.values[0] - top2.values[1]))
        out.append(nxt)
        if (stop_on_eos and nxt in eos_ids) or step == max_new_tokens - 1:
            break
        logits, cache = decode_step(model, nxt, cache)
        cache_lens.append(cache.get_seq_length())
    _sync(dev)
    total = time.perf_counter() - t0
    # sanity: cache grew by exactly one position per decode step
    assert cache_lens == list(range(len(prompt_ids), len(prompt_ids) + len(cache_lens)))
    return {"tokens": out, "margins": margins, "prefill_s": t_prefill,
            "decode_s": total - t_prefill, "total_s": total,
            "final_cache_len": cache_lens[-1], "cache_bytes": cache_bytes(cache)}


def cache_bytes(cache: DynamicCache) -> int:
    return int(sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                   for k, v in zip(cache.key_cache, cache.value_cache)))


def compare_paths(a: List[int], b: List[int], margins_ref: List[float]) -> Dict:
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    if first is None and len(a) != len(b):
        first = n
    return {"equal": first is None, "first_mismatch": first,
            "margin_at_mismatch": (margins_ref[first] if first is not None and first < len(margins_ref)
                                   else None)}
