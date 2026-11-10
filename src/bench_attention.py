"""Task 4 validation: reference SDPA vs. tiled online-softmax vs. PyTorch fused SDPA.

Correctness: several shapes (incl. non-multiple-of-block lengths, Lq != Lk decode-style
causal masks, fp32 and fp16), max/mean absolute error vs. the float32 reference.
Performance: median time and peak memory (CUDA) for growing sequence length.

Usage: python src/bench_attention.py
"""
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS_DIR, write_json
from tiled_attention import reference_attention, tiled_attention


def timed(fn, dev, reps=5):
    fn()
    ts = []
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for _ in range(reps):
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    mem = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else None
    return statistics.median(ts) * 1000, mem


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(0)
    correctness = []
    cases = [  # B, H, Lq, Lk, D, causal, dtype, block
        (1, 1, 4, 4, 8, True, torch.float32, 2),
        (2, 4, 37, 37, 16, True, torch.float32, 8),
        (2, 4, 37, 37, 16, False, torch.float32, 8),
        (1, 8, 128, 128, 64, True, torch.float32, 32),
        (1, 8, 1, 200, 64, True, torch.float32, 64),     # single-token decode step
        (1, 8, 5, 133, 64, True, torch.float32, 16),     # speculative verify block
        (2, 12, 256, 256, 64, True, torch.float16, 64),
        (1, 12, 512, 512, 128, True, torch.float16, 128),
    ]
    for B, H, Lq, Lk, D, causal, dt, blk in cases:
        q = torch.randn(B, H, Lq, D, generator=g).to(dev, dt)
        k = torch.randn(B, H, Lk, D, generator=g).to(dev, dt)
        v = torch.randn(B, H, Lk, D, generator=g).to(dev, dt)
        ref = reference_attention(q.float(), k.float(), v.float(), causal=causal)
        til = tiled_attention(q, k, v, causal=causal, block_q=blk, block_k=blk).float()
        err = (til - ref).abs()
        tol = 1e-5 if dt == torch.float32 else 2e-3
        row = {"shape": [B, H, Lq, Lk, D], "causal": causal, "dtype": str(dt).split(".")[-1],
               "block": blk, "max_abs_err": err.max().item(), "mean_abs_err": err.mean().item(),
               "tolerance": tol, "pass": bool(torch.allclose(til, ref, atol=tol, rtol=tol))}
        if Lq == Lk:  # fused backend uses top-left causal alignment, identical when Lq == Lk
            fused = F.scaled_dot_product_attention(q, k, v, is_causal=causal).float()
            row["fused_max_abs_err"] = (fused - ref).abs().max().item()
        correctness.append(row)
        print(row, flush=True)

    perf = []
    dt = torch.float16 if dev == "cuda" else torch.float32
    for L in [256, 512, 1024, 2048] + ([4096] if dev == "cuda" else []):
        q, k, v = (torch.randn(1, 12, L, 64, generator=g).to(dev, dt) for _ in range(3))
        row = {"seq_len": L, "heads": 12, "head_dim": 64, "dtype": str(dt).split(".")[-1]}
        for name, fn in [("reference", lambda: reference_attention(q, k, v, True)),
                         ("tiled_b128", lambda: tiled_attention(q, k, v, True, 128, 128)),
                         ("torch_sdpa_fused", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))]:
            try:
                ms, mem = timed(fn, dev)
                row[name] = {"ms": ms, "peak_mem_mb": mem}
            except torch.cuda.OutOfMemoryError:
                row[name] = {"ms": None, "peak_mem_mb": "OOM"}
                torch.cuda.empty_cache()
        perf.append(row)
        print(row, flush=True)
    write_json({"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                "correctness": correctness, "performance": perf,
                "all_pass": all(r["pass"] for r in correctness)},
               os.path.join(RESULTS_DIR, "bench_attention.json"))


if __name__ == "__main__":
    main()
