"""Task 3 benchmark: KV-cached vs. no-cache greedy decoding.

For several (input length, output length) settings, on the SAME model and inputs:
 - token-level agreement (first mismatch + logit margin if any)
 - prefill time, decode time, output tokens/s, peak GPU memory, KV-cache size
Protocol: 1 warm-up run, then R synchronised repeats; median reported.
EOS stopping is disabled so exactly ``out_len`` tokens are produced.

Usage: python src/bench_kv.py [--adapter outputs/adapter] [--repeats 3]
"""
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_DIR, RESULTS_DIR, TARGET_MODEL, build_prompt_ids, load_model, read_jsonl, write_json
from kv_cache import compare_paths, greedy_kv_cache, greedy_no_cache


def make_prompt(tok, rec, target_len):
    """Prompt from a validation record, padded/trimmed to ~target_len tokens by
    repeating note sentences (benchmark only; never used for evaluation)."""
    sents = {int(k): v for k, v in rec["sentences"].items()}
    ids, _ = build_prompt_ids(tok, sents, rec["criterion_type"], rec["criterion_text"], max_tokens=10**6)
    filler = list(sents.values())
    j = 0
    while len(ids) < target_len:
        sents[len(sents)] = filler[j % len(filler)]
        j += 1
        ids, _ = build_prompt_ids(tok, sents, rec["criterion_type"], rec["criterion_text"], max_tokens=10**6)
    return ids


def peak(fn, dev):
    if dev == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    r = fn()
    r["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TARGET_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--settings", default="256x32,512x64,1024x128")
    args = ap.parse_args()
    model, tok = load_model(args.model, args.adapter)
    dev = model.device.type
    rec = read_jsonl(os.path.join(DATA_DIR, "processed", "val.jsonl"))[0]
    rows = []
    for s in args.settings.split(","):
        in_len, out_len = map(int, s.split("x"))
        ids = make_prompt(tok, rec, in_len)
        greedy_kv_cache(model, ids, 4, stop_on_eos=False)          # warm-up
        greedy_no_cache(model, ids, 4, stop_on_eos=False)
        res = {"kv": [], "nocache": []}
        for _ in range(args.repeats):
            res["kv"].append(peak(lambda: greedy_kv_cache(model, ids, out_len, stop_on_eos=False), dev))
            res["nocache"].append(peak(lambda: greedy_no_cache(model, ids, out_len, stop_on_eos=False), dev))
        cmp = compare_paths(res["kv"][0]["tokens"], res["nocache"][0]["tokens"], res["nocache"][0]["margins"])
        row = {"input_tokens": len(ids), "output_tokens": out_len, **cmp,
               "min_margin_kv": min(res["kv"][0]["margins"])}
        for mode in ["kv", "nocache"]:
            med = lambda key: statistics.median(r[key] for r in res[mode])
            row[mode] = {"prefill_s": med("prefill_s"), "decode_s": med("decode_s"),
                         "total_s": med("total_s"),
                         "decode_tok_per_s": (out_len - 1) / med("decode_s"),
                         "e2e_tok_per_s": out_len / med("total_s"),
                         "peak_mem_mb": max((r["peak_mem_mb"] or 0) for r in res[mode]) or None}
        row["kv"]["cache_mb"] = res["kv"][0]["cache_bytes"] / 2**20
        row["speedup_total"] = row["nocache"]["total_s"] / row["kv"]["total_s"]
        rows.append(row)
        print(f"{len(ids)}x{out_len}: equal={cmp['equal']} kv={row['kv']['total_s']:.2f}s "
              f"nocache={row['nocache']['total_s']:.2f}s speedup={row['speedup_total']:.2f}x", flush=True)
    write_json({"model": args.model, "adapter": args.adapter, "device": dev,
                "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                "dtype": str(model.dtype), "repeats": args.repeats, "warmup_runs": 1,
                "timing": "time.perf_counter with torch.cuda.synchronize; median of repeats",
                "results": rows}, os.path.join(RESULTS_DIR, "bench_kv.json"))


if __name__ == "__main__":
    main()
