"""Task 5 — Greedy speculative decoding (Leviathan et al. 2023; Chen et al. 2023),
implemented by hand on top of explicit KV caches (see kv_cache.py).

Per iteration, with committed sequence x_1..x_n (target cache holds K/V of x_1..x_{n-1};
``t_next`` = the target's greedy choice for position n+1 is already known from its logits):
 1. DRAFT: the small model proposes k tokens d_1..d_k greedily *after* t_next (t_next is
    always correct because it came from the target, so we feed it to the draft first).
 2. VERIFY: the target runs ONE forward pass over [t_next, d_1, ..., d_k] (k+1 tokens,
    positions n..n+k, causal mask inside the block + full view of the cache). This yields
    the target's greedy prediction g_0..g_k after each prefix.
 3. ACCEPT: the longest prefix with d_i == g_{i-1} is accepted (a accepted drafts).
 4. CORRECT: the target's own token g_a is the next "known-correct" token (replacement
    on rejection, bonus token on full acceptance), so every iteration commits a+1 tokens.
 5. CACHE SYNC: both caches are cropped back to the committed length, discarding K/V of
    rejected draft tokens, so the next iteration starts from a consistent state.
Greedy acceptance guarantees the output is token-identical to plain greedy decoding with
the target (up to fp16 tie-breaking in batched vs. single-token forward passes).
"""
from __future__ import annotations

import time
from typing import Dict, List

import torch
from transformers import DynamicCache

from kv_cache import _sync, decode_step, prefill


def crop_cache(cache: DynamicCache, length: int) -> None:
    """Discard cached K/V beyond ``length`` positions (rejected speculative tokens)."""
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][..., :length, :]
        cache.value_cache[i] = cache.value_cache[i][..., :length, :]
    cache._seen_tokens = length


@torch.no_grad()
def _forward_block(model, tokens: List[int], cache: DynamicCache):
    """Feed several tokens at once at positions [past, past+len). The model builds a
    causal mask over the new block while letting each query see the whole cache."""
    dev = model.device
    past = cache.get_seq_length()
    ids = torch.tensor([tokens], device=dev)
    pos = torch.arange(past, past + len(tokens), device=dev)
    mask = torch.ones((1, past + len(tokens)), dtype=torch.long, device=dev)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos.unsqueeze(0),
                cache_position=pos, past_key_values=cache, use_cache=True)
    return out.logits[0].float()   # [len, vocab]


@torch.no_grad()
def speculative_greedy(target, draft, prompt_ids: List[int], max_new_tokens: int,
                       k: int = 4, eos_ids=()) -> Dict:
    t_dev, d_dev = target.device, draft.device
    _sync(t_dev); _sync(d_dev)
    t0 = time.perf_counter()
    t_logits, t_cache = prefill(target, prompt_ids)
    _, d_cache = prefill(draft, prompt_ids[:-1]) if len(prompt_ids) > 1 else (None, DynamicCache())
    # invariant: d_cache holds K/V for committed[:-1]; t_cache likewise; ``nxt`` is the
    # target-verified token that follows ``committed``.
    committed = list(prompt_ids)
    d_pending = [prompt_ids[-1]]           # tokens committed but not yet in the draft cache
    nxt = int(torch.argmax(t_logits))
    out: List[int] = []
    stats = {"iterations": 0, "drafted": 0, "accepted": 0, "accepted_per_iter": []}
    _sync(t_dev)
    t_prefill = time.perf_counter() - t0
    t_draft = t_verify = 0.0

    while True:
        out.append(nxt)
        if nxt in eos_ids or len(out) >= max_new_tokens:
            break
        kk = min(k, max_new_tokens - len(out))   # never draft past the budget
        # ---- 1. draft k tokens after nxt
        ts = time.perf_counter()
        d_logits = _forward_block(draft, d_pending + [nxt], d_cache)[-1]
        drafts = []
        for i in range(kk):
            d = int(torch.argmax(d_logits))
            drafts.append(d)
            if d in eos_ids or i == kk - 1:
                break
            d_logits, d_cache = decode_step(draft, d, d_cache)
        _sync(d_dev)
        t_draft += time.perf_counter() - ts
        # ---- 2. verify [nxt, d_1..d_k] with ONE target pass
        ts = time.perf_counter()
        v_logits = _forward_block(target, [nxt] + drafts, t_cache)   # [k+1, V]
        g = torch.argmax(v_logits, dim=-1).tolist()                  # g[i] = target after drafts[:i]
        _sync(t_dev)
        t_verify += time.perf_counter() - ts
        # ---- 3. accept longest matching prefix
        a = 0
        while a < len(drafts) and drafts[a] == g[a]:
            a += 1
        accepted = drafts[:a]
        stats["iterations"] += 1
        stats["drafted"] += len(drafts)
        stats["accepted"] += a
        stats["accepted_per_iter"].append(a)
        # ---- 4. commit nxt + accepted drafts; correction/bonus token = g[a]
        committed += [nxt] + accepted
        stop = False
        for tok in accepted:
            out.append(tok)
            if tok in eos_ids or len(out) >= max_new_tokens:
                stop = True
                break
        if stop:
            break
        nxt = int(g[a])
        # ---- 5. cache synchronisation: keep K/V only for committed tokens
        crop_cache(t_cache, len(committed))
        # draft cache contains K/V for committed[:-?]: everything it processed that is
        # still committed. It processed (old committed) + nxt + drafts[:-1].
        d_len = min(d_cache.get_seq_length(), len(committed))
        crop_cache(d_cache, d_len)
        d_pending = committed[d_len:]
        assert t_cache.get_seq_length() == len(committed)

    _sync(t_dev)
    total = time.perf_counter() - t0
    stats["acceptance_rate"] = stats["accepted"] / max(1, stats["drafted"])
    stats["mean_accepted_per_iter"] = stats["accepted"] / max(1, stats["iterations"])
    stats["tokens_per_target_pass"] = len(out) / max(1, stats["iterations"] + 1)
    return {"tokens": out, "prefill_s": t_prefill, "draft_s": t_draft, "verify_s": t_verify,
            "total_s": total, "stats": stats}
