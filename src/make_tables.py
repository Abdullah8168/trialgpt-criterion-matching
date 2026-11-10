"""Write LaTeX tables (report/tables/*.tex) from results/*.json so every reported number
comes straight from the logged experiment outputs."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_DIR, RESULTS_DIR, ROOT

OUT = os.path.join(ROOT, "report", "tables")
os.makedirs(OUT, exist_ok=True)


def load(name, base=RESULTS_DIR):
    p = os.path.join(base, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, d=3):
    return "--" if x is None else f"{x:.{d}f}"


def write(name, body):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write(body)


def esc(s):
    return s.replace("_", r"\_").replace("&", r"\&")


def main():
    m = load("split_manifest.json", DATA_DIR)
    if m:
        rows = []
        for s in ["train", "val", "test"]:
            v = m["splits"][s]
            rows.append(f"{s} & {v['n_patients']} & {v['n_annotations']} & {v['n_inclusion']} & "
                        f"{v['n_exclusion']} & {v['n_with_gold_evidence']} \\\\")
        write("split.tex", "\n".join(rows))

    # main before/after table
    rows = []
    for split in ["val", "test"]:
        for tag in ["base", "finetuned"]:
            r = load(f"eval_{split}_{tag}.json")
            if not r:
                continue
            o, e = r["overall"], r["overall"]["evidence"]
            rows.append(f"{split} & {'zero-shot' if tag == 'base' else 'LoRA'} & {f(o['accuracy'])} & "
                        f"{f(o['macro_f1'])} & {f(r['inclusion']['accuracy'])} & {f(r['inclusion']['macro_f1'])} & "
                        f"{f(r['exclusion']['accuracy'])} & {f(r['exclusion']['macro_f1'])} & {r['parse_errors']} \\\\")
    write("before_after.tex", "\n".join(rows))

    # evidence table
    rows = []
    for split in ["val", "test"]:
        for tag in ["base", "finetuned"]:
            r = load(f"eval_{split}_{tag}.json")
            if not r:
                continue
            e = r["overall"]["evidence"]
            rows.append(f"{split} & {'zero-shot' if tag == 'base' else 'LoRA'} & {f(e['micro_precision'])} & "
                        f"{f(e['micro_recall'])} & {f(e['micro_f1'])} & {f(e['macro_precision'])} & "
                        f"{f(e['macro_recall'])} & {f(e['macro_f1'])} & {f(e['empty_gold_agreement'])} \\\\")
    write("evidence.tex", "\n".join(rows))

    # per-class test table (fine-tuned)
    r = load("eval_test_finetuned.json")
    if r:
        rows = []
        for part in ["inclusion", "exclusion"]:
            for lab, v in r[part]["per_class"].items():
                rows.append(f"{part} & {lab} & {f(v['precision'])} & {f(v['recall'])} & {f(v['f1'])} & {v['support']} \\\\")
        write("per_class_test.tex", "\n".join(rows))

    t = load("train_log_adapter.json")
    if t:
        h = t["hyperparameters"]
        write("train_cfg.tex", "\n".join([
            f"Base model & {esc(t['model'])} \\\\",
            f"Method & LoRA, r={h['rank']}, $\\alpha$={h['alpha']}, dropout={h['dropout']} \\\\",
            "Target modules & q,k,v,o,gate,up,down projections \\\\",
            f"Trainable / total params & {t['trainable_params']/1e6:.2f}M / {t['total_params']/1e9:.3f}B \\\\",
            f"Optimizer / LR & AdamW, {h['lr']}, cosine, 5\\% warm-up \\\\",
            f"Batch & {h['bsz']} $\\times$ {h['grad_accum']} accumulation = {h['bsz']*h['grad_accum']} \\\\",
            f"Epochs & {h['epochs']} (best val-loss checkpoint) \\\\",
            f"Precision & {esc(t['precision'])} \\\\",
            f"Hardware & {esc(t['gpu'])}, peak {f(t['peak_gpu_mem_gb'],1)} GB \\\\",
            f"Training time & {t['train_time_s']/60:.1f} min \\\\",
            f"Best val loss & {f(t['best_eval_loss'],4)} \\\\"]))

    kv = load("bench_kv.json")
    if kv:
        rows = []
        for x in kv["results"]:
            for mode in ["nocache", "kv"]:
                y = x[mode]
                rows.append(f"{x['input_tokens']}/{x['output_tokens']} & {'no cache' if mode == 'nocache' else 'KV cache'} & "
                            f"{y['prefill_s']*1000:.1f} & {y['decode_s']:.2f} & {y['decode_tok_per_s']:.1f} & "
                            f"{f(y['peak_mem_mb'],0)} & {'yes' if x['equal'] else 'no@'+str(x['first_mismatch'])} \\\\")
        write("kv.tex", "\n".join(rows))

    at = load("bench_attention.json")
    if at:
        X = "$\\times$"
        rows = [f"{X.join(map(str, c['shape']))} &{'C' if c['causal'] else 'F'} & {c['dtype']} & {c['block']} & "
                f"{c['max_abs_err']:.1e} & {c['mean_abs_err']:.1e} & {'pass' if c['pass'] else 'FAIL'} \\\\"
                for c in at["correctness"]]
        write("attn_correct.tex", "\n".join(rows))
        rows = []
        for p in at["performance"]:
            cells = [str(p["seq_len"])]
            for n in ["reference", "tiled_b128", "torch_sdpa_fused"]:
                cells += [f(p[n]["ms"], 2), f(p[n]["peak_mem_mb"], 0) if isinstance(p[n]["peak_mem_mb"], float) else str(p[n]["peak_mem_mb"])]
            rows.append(" & ".join(cells) + " \\\\")
        write("attn_perf.tex", "\n".join(rows))

    sp = load("bench_speculative.json")
    if sp:
        g = sp["greedy_target"]
        rows = [f"greedy target & -- & -- & -- & {g['total_s']:.2f} & {g['tok_per_s']:.1f} & 1.00 & {f(g['peak_mem_mb'],0)} & ref \\\\"]
        for x in sp["speculative"]:
            rows.append(f"{esc(x['draft'])} & {x['k']} & {x['acceptance_rate']:.2f} & {x['mean_accepted_per_iter']:.2f} & "
                        f"{x['total_s']:.2f} & {x['tok_per_s']:.1f} & {x['speedup']:.2f} & {f(x['peak_mem_mb'],0)} & "
                        f"{x['token_equal_prompts']} \\\\")
        write("spec.tex", "\n".join(rows))
    print("tables written to", OUT, os.listdir(OUT))


if __name__ == "__main__":
    main()
