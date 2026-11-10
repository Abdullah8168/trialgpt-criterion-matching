"""Task 5 benchmark: greedy speculative decoding vs. plain greedy (KV-cached) target decoding.

Same prompts (validation records), same target. Reports token equality, acceptance rate,
latency, tokens/s, speedup/slowdown and peak memory for several draft lengths k, and for
the base vs. fine-tuned draft model (if a draft adapter is given).

Usage: python src/bench_speculative.py --adapter outputs/adapter [--draft_adapter outputs/draft_adapter]
"""
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (DATA_DIR, DRAFT_MODEL, RESULTS_DIR, TARGET_MODEL, build_prompt_ids, load_model,
                    read_jsonl, write_json)
from kv_cache import greedy_kv_cache
from speculative import speculative_greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TARGET_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--draft", default=DRAFT_MODEL)
    ap.add_argument("--draft_adapter", default=None)
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--ks", default="2,4,6")
    ap.add_argument("--draft_device", default=None)
    args = ap.parse_args()

    target, tok = load_model(args.model, args.adapter)
    drafts = {"draft_base": load_model(args.draft, None, device=args.draft_device)[0]}
    if args.draft_adapter:
        drafts["draft_finetuned"] = load_model(args.draft, args.draft_adapter, device=args.draft_device)[0]
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    recs = read_jsonl(os.path.join(DATA_DIR, "processed", "val.jsonl"))[: args.n_prompts]
    prompts = []
    for r in recs:
        s = {int(k): v for k, v in r["sentences"].items()}
        prompts.append(build_prompt_ids(tok, s, r["criterion_type"], r["criterion_text"])[0])

    dev = target.device.type
    def mem_reset():
        if dev == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    def mem_peak():
        return torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else None

    # warm-up
    greedy_kv_cache(target, prompts[0], 8, eos)
    for d in drafts.values():
        speculative_greedy(target, d, prompts[0], 8, k=2, eos_ids=eos)

    mem_reset()
    base = [greedy_kv_cache(target, p, args.max_new, eos) for p in prompts]
    base_mem = mem_peak()
    base_time = sum(b["total_s"] for b in base)
    base_tokens = sum(len(b["tokens"]) for b in base)
    results = {"model": args.model, "draft": args.draft, "device": dev,
               "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
               "n_prompts": len(prompts), "max_new_tokens": args.max_new,
               "greedy_target": {"total_s": base_time, "tokens": base_tokens,
                                 "tok_per_s": base_tokens / base_time, "peak_mem_mb": base_mem},
               "speculative": []}
    print("greedy:", results["greedy_target"], flush=True)
    for dname, dmodel in drafts.items():
        for k in map(int, args.ks.split(",")):
            mem_reset()
            outs = [speculative_greedy(target, dmodel, p, args.max_new, k=k, eos_ids=eos) for p in prompts]
            mem = mem_peak()
            eq = [o["tokens"] == b["tokens"] for o, b in zip(outs, base)]
            t = sum(o["total_s"] for o in outs)
            n = sum(len(o["tokens"]) for o in outs)
            drafted = sum(o["stats"]["drafted"] for o in outs)
            acc = sum(o["stats"]["accepted"] for o in outs)
            row = {"draft": dname, "k": k, "token_equal_prompts": f"{sum(eq)}/{len(eq)}",
                   "all_equal": all(eq), "acceptance_rate": acc / max(1, drafted),
                   "mean_accepted_per_iter": statistics.mean(o["stats"]["mean_accepted_per_iter"] for o in outs),
                   "total_s": t, "tok_per_s": n / t, "speedup": base_time / t,
                   "draft_time_s": sum(o["draft_s"] for o in outs),
                   "verify_time_s": sum(o["verify_s"] for o in outs), "peak_mem_mb": mem}
            results["speculative"].append(row)
            print(row, flush=True)
    write_json(results, os.path.join(RESULTS_DIR, "bench_speculative.json"))


if __name__ == "__main__":
    main()
