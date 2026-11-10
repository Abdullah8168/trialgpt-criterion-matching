"""Generate report figures + LaTeX-ready tables from results/*.json (skips missing files)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS_DIR

FIG = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG, exist_ok=True)


def load(name):
    p = os.path.join(RESULTS_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def confusion(rep, tag):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, part in zip(axes, ["inclusion", "exclusion"]):
        cm = np.array(rep[part]["confusion_matrix"]["matrix"])
        labs = rep[part]["confusion_matrix"]["labels"]
        ax.imshow(cm, cmap="Blues")
        short = [l.replace("not enough information", "NEI").replace("not applicable", "N/A") for l in labs]
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(short, fontsize=8)
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(f"{part} ({tag})")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"confusion_{tag}.png"), dpi=200); plt.close()


def main():
    for split in ["val", "test"]:
        for tag in ["base", "finetuned"]:
            r = load(f"eval_{split}_{tag}.json")
            if r:
                confusion(r, f"{split}_{tag}")
    # before/after bars
    bars = {}
    for split in ["val", "test"]:
        for tag in ["base", "finetuned"]:
            r = load(f"eval_{split}_{tag}.json")
            if r:
                bars[f"{split}-{tag}"] = [r["overall"]["accuracy"], r["overall"]["macro_f1"],
                                          r["overall"]["evidence"]["micro_f1"]]
    if bars:
        x = np.arange(3); w = 0.8 / len(bars)
        plt.figure(figsize=(6, 3.2))
        for i, (k, v) in enumerate(bars.items()):
            plt.bar(x + i * w, v, w, label=k)
        plt.xticks(x + w * (len(bars) - 1) / 2, ["accuracy", "macro-F1", "evidence micro-F1"])
        plt.ylim(0, 1); plt.legend(fontsize=8); plt.title("Before vs. after fine-tuning"); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "before_after.png"), dpi=200); plt.close()
    kv = load("bench_kv.json")
    if kv:
        rs = kv["results"]
        lab = [f"{r['input_tokens']}/{r['output_tokens']}" for r in rs]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        x = np.arange(len(rs))
        ax[0].bar(x - 0.2, [r["kv"]["total_s"] for r in rs], 0.4, label="KV cache")
        ax[0].bar(x + 0.2, [r["nocache"]["total_s"] for r in rs], 0.4, label="no cache")
        ax[0].set_xticks(x); ax[0].set_xticklabels(lab); ax[0].set_xlabel("input/output tokens")
        ax[0].set_ylabel("latency (s)"); ax[0].legend()
        ax[1].bar(x - 0.2, [r["kv"]["peak_mem_mb"] or 0 for r in rs], 0.4, label="KV cache")
        ax[1].bar(x + 0.2, [r["nocache"]["peak_mem_mb"] or 0 for r in rs], 0.4, label="no cache")
        ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].set_ylabel("peak GPU memory (MB)")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "kv_bench.png"), dpi=200); plt.close()
    at = load("bench_attention.json")
    if at:
        rs = at["performance"]
        plt.figure(figsize=(5, 3.2))
        for name in ["reference", "tiled_b128", "torch_sdpa_fused"]:
            pts = [(r["seq_len"], r[name]["ms"]) for r in rs if r[name]["ms"]]
            plt.plot(*zip(*pts), "o-", label=name)
        plt.xscale("log", base=2); plt.yscale("log"); plt.xlabel("sequence length"); plt.ylabel("ms")
        plt.legend(); plt.title("Causal attention runtime"); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "attention_bench.png"), dpi=200); plt.close()
    sp = load("bench_speculative.json")
    if sp:
        plt.figure(figsize=(5, 3.2))
        for d in sorted({r["draft"] for r in sp["speculative"]}):
            pts = [(r["k"], r["speedup"]) for r in sp["speculative"] if r["draft"] == d]
            plt.plot(*zip(*pts), "o-", label=d)
        plt.axhline(1, color="gray", ls="--"); plt.xlabel("draft length k"); plt.ylabel("speedup vs greedy")
        plt.legend(); plt.title("Speculative decoding"); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "spec_bench.png"), dpi=200); plt.close()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
