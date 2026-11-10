"""Task 6 — Browser-based clinical-trial criterion screening application (Gradio).

Start:  python app.py      (or double-click start_app.bat)  ->  http://127.0.0.1:7860
Env:    ADAPTER_DIR (default outputs/adapter), DRAFT_ADAPTER_DIR (default outputs/draft_adapter),
        TARGET_MODEL, DRAFT_MODEL, DRAFT_DEVICE (auto|cuda|cpu)

EDUCATIONAL SCREENING PROTOTYPE — NOT A CLINICAL ENROLLMENT DECISION TOOL.
"""
import datetime as dt
import hashlib
import html
import json
import os
import sys
import tempfile
import threading
import time
import traceback

import gradio as gr
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import (DATA_DIR, DRAFT_MODEL, LABELS, MAX_PROMPT_TOKENS, RESULTS_DIR, TARGET_MODEL,
                    build_prompt_ids, explain, load_model, parse_output, read_jsonl, split_free_text)
from kv_cache import greedy_kv_cache, greedy_no_cache, compare_paths
from speculative import speculative_greedy
from tiled_attention import reference_attention, tiled_attention

DISCLAIMER = ("⚠️ **Educational screening prototype — NOT a clinical enrollment decision.** "
              "Outputs are model predictions on individual criteria and may be wrong. A qualified "
              "clinician/study coordinator must review every criterion.")

ADAPTER = os.environ.get("ADAPTER_DIR", os.path.join(ROOT, "outputs", "adapter"))
DRAFT_ADAPTER = os.environ.get("DRAFT_ADAPTER_DIR", os.path.join(ROOT, "outputs", "draft_adapter"))
MAX_CRITERIA = 40
MAX_NEW_TOKENS = 48

_state = {"target": None, "tok": None, "draft": None, "lock": threading.Lock(), "info": {}}


def get_target():
    if _state["target"] is None:
        ad = ADAPTER if os.path.isdir(ADAPTER) else None
        m, t = load_model(TARGET_MODEL, ad)
        _state.update(target=m, tok=t)
        _state["info"].update(target=TARGET_MODEL, adapter=ad or "NONE (zero-shot base model)",
                              target_device=str(m.device), dtype=str(m.dtype))
    return _state["target"], _state["tok"]


def get_draft():
    if _state["draft"] is None:
        get_target()
        dev = os.environ.get("DRAFT_DEVICE", "auto")
        if dev == "auto":
            dev = "cpu"
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                dev = "cuda" if free > 1.6 * 2**30 else "cpu"
        ad = DRAFT_ADAPTER if os.path.isdir(DRAFT_ADAPTER) else None
        _state["draft"] = load_model(DRAFT_MODEL, ad, device=dev)[0]
        _state["info"].update(draft=DRAFT_MODEL, draft_adapter=ad or "NONE", draft_device=dev)
    return _state["draft"]


def eos_ids(tok):
    return {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}


# ----------------------------------------------------------------------------- screening
def parse_criteria(inc_text, exc_text):
    crit = []
    for ctype, text in [("inclusion", inc_text or ""), ("exclusion", exc_text or "")]:
        for line in text.splitlines():
            line = line.strip().lstrip("-•*").strip()
            if len(line) >= 3:
                crit.append((ctype, line))
    return crit


def decode(mode, model, tok, ids):
    eos = eos_ids(tok)
    if mode == "Speculative (draft+target)":
        r = speculative_greedy(model, get_draft(), ids, MAX_NEW_TOKENS, k=4, eos_ids=eos)
    elif mode == "No cache (reference)":
        r = greedy_no_cache(model, ids, MAX_NEW_TOKENS, eos_ids=eos)
    else:
        r = greedy_kv_cache(model, ids, MAX_NEW_TOKENS, eos_ids=eos)
    return r


def aggregate(results):
    """Explicit rule-based aggregation (NOT a ground-truth-labelled outcome):
    1) any inclusion 'not included' or exclusion 'excluded'  -> LIKELY INELIGIBLE
    2) else any 'not enough information'                     -> UNDETERMINED (needs info)
    3) else (all met / not excluded / not applicable)        -> POTENTIALLY ELIGIBLE"""
    fails = [r for r in results if r["label"] in ("not included", "excluded")]
    unknown = [r for r in results if r["label"] == "not enough information"]
    if fails:
        return "LIKELY INELIGIBLE", fails, unknown
    if unknown:
        return "UNDETERMINED — more information needed", fails, unknown
    return "POTENTIALLY ELIGIBLE (pending clinician review)", fails, unknown


COLORS = {"included": "#1a7f37", "not excluded": "#1a7f37", "not included": "#cf222e",
          "excluded": "#cf222e", "not enough information": "#9a6700", "not applicable": "#57606a"}


def render_html(note_sents, results, verdict, fails, unknown, meta):
    rows = []
    for i, r in enumerate(results, 1):
        ev = "".join(f"<li><b>[{k}]</b> {html.escape(v)}</li>" for k, v in r["evidence_text"].items()) \
            or "<li><i>no sentence cited</i></li>"
        warn = " ⚠️ output could not be parsed; defaulted to 'not enough information'" if r["parse_error"] else ""
        rows.append(
            f"<div style='border:1px solid #d0d7de;border-radius:8px;padding:10px;margin:8px 0'>"
            f"<div><b>#{i} [{r['criterion_type'].upper()}]</b> {html.escape(r['criterion'])}</div>"
            f"<div style='margin:6px 0'><span style='background:{COLORS[r['label']]};color:#fff;"
            f"padding:2px 8px;border-radius:10px'>{r['label']}</span>{warn}</div>"
            f"<div><b>Evidence sentences:</b><ul style='margin:2px 0'>{ev}</ul></div>"
            f"<div><b>Explanation:</b> {html.escape(r['explanation'])}</div>"
            f"<div style='color:#57606a;font-size:12px'>raw model output: <code>{html.escape(r['raw'])}</code>"
            f" · {r['latency_s']:.2f}s</div></div>")
    missing = "".join(f"<li>{html.escape(r['criterion'])}</li>" for r in unknown) or "<li>none</li>"
    note = "".join(f"<li value='{k}'>{html.escape(v)}</li>" for k, v in sorted(note_sents.items()))
    prov = "".join(f"<li><b>{k}</b>: {html.escape(str(v))}</li>" for k, v in meta.items())
    return f"""
<div style='font-family:sans-serif'>
<div style='background:#fff8c5;border:1px solid #d4a72c;padding:8px;border-radius:6px'>
<b>Educational screening prototype — not a clinical enrollment decision.</b></div>
<h3>Preliminary aggregated outcome: {verdict}</h3>
<p style='font-size:12px;color:#57606a'>Rule: any inclusion criterion 'not included' or exclusion
criterion 'excluded' ⇒ likely ineligible; otherwise any 'not enough information' ⇒ undetermined;
otherwise potentially eligible. This is an aggregation of criterion-level predictions, not a
ground-truth patient–trial label.</p>
<p><b>Criteria failed:</b> {len(fails)} · <b>Missing information:</b> {len(unknown)} ·
<b>Total criteria:</b> {len(results)}</p>
<h4>Information needed</h4><ul>{missing}</ul>
<h4>Criterion-level results</h4>{''.join(rows)}
<h4>Patient record (sentence IDs)</h4><ol start='0'>{note}</ol>
<h4>Provenance</h4><ul>{prov}</ul></div>"""


def screen(note, inc, exc, mode, progress=gr.Progress()):
    try:
        if not note or len(note.strip()) < 20:
            raise gr.Error("Please enter a patient record (at least 20 characters).")
        crit = parse_criteria(inc, exc)
        if not crit:
            raise gr.Error("Please enter at least one inclusion or exclusion criterion (one per line).")
        if len(crit) > MAX_CRITERIA:
            raise gr.Error(f"Too many criteria ({len(crit)}); the limit is {MAX_CRITERIA}.")
        sents = split_free_text(note)
        if not sents:
            raise gr.Error("Could not find any sentences in the patient record.")
        with _state["lock"]:
            progress(0, desc="Loading model (first run can take a minute)...")
            model, tok = get_target()
            if mode == "Speculative (draft+target)":
                get_draft()
            results, table, warnings = [], [], []
            t_all = time.time()
            for i, (ctype, ctext) in enumerate(crit):
                progress((i + 1) / len(crit), desc=f"Criterion {i + 1}/{len(crit)}")
                ids, dropped = build_prompt_ids(tok, sents, ctype, ctext)
                if dropped:
                    warnings.append(f"Criterion {i + 1}: prompt exceeded {MAX_PROMPT_TOKENS} tokens; "
                                    f"the last {dropped} note sentence(s) were truncated.")
                r = decode(mode, model, tok, ids)
                text = tok.decode(r["tokens"], skip_special_tokens=True)
                p = parse_output(text, ctype, set(sents))
                p.update(criterion_type=ctype, criterion=ctext,
                         evidence_text={k: sents[k] for k in p["evidence"]},
                         explanation=explain(p["label"], ctype, p["evidence"], sents),
                         latency_s=r["total_s"])
                results.append(p)
                table.append([i + 1, ctype, ctext[:90], p["label"], ", ".join(map(str, p["evidence"])) or "—",
                              f"{r['total_s']:.2f}"])
        verdict, fails, unknown = aggregate(results)
        meta = {"generated": dt.datetime.now().isoformat(timespec="seconds"), **_state["info"],
                "decoding": f"{mode}, greedy, max_new_tokens={MAX_NEW_TOKENS}",
                "note_sha256": hashlib.sha256(note.encode()).hexdigest()[:16],
                "total_runtime_s": round(time.time() - t_all, 2)}
        page = render_html(sents, results, verdict, fails, unknown, meta)
        path = os.path.join(tempfile.gettempdir(), f"screening_report_{int(time.time())}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"<html><head><meta charset='utf-8'><title>Eligibility screening report</title></head>"
                    f"<body style='max-width:900px;margin:auto'>{page}</body></html>")
        jpath = path.replace(".html", ".json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump({"provenance": meta, "aggregated_outcome": verdict, "sentences": sents,
                       "results": results}, f, indent=2)
        status = f"✅ Screened {len(results)} criteria → **{verdict}**"
        if warnings:
            status += "\n\n" + "\n".join("⚠️ " + w for w in warnings)
        return status, table, page, [path, jpath]
    except gr.Error:
        raise
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise gr.Error("GPU out of memory. Try fewer criteria, a shorter note, or the KV-cache mode.")
    except Exception as e:  # never crash the UI
        traceback.print_exc()
        raise gr.Error(f"Unexpected error: {type(e).__name__}: {e}")


def load_example(idx):
    recs = read_jsonl(os.path.join(DATA_DIR, "processed", "val.jsonl"))  # validation, never test
    pids = sorted({r["patient_id"] for r in recs})
    pid = pids[int(idx) % len(pids)]
    rs = [r for r in recs if r["patient_id"] == pid]
    note = "\n".join(f"{k}. {v}" for k, v in sorted(rs[0]["sentences"].items(), key=lambda x: int(x[0])))
    inc = "\n".join(r["criterion_text"] for r in rs if r["criterion_type"] == "inclusion")
    exc = "\n".join(r["criterion_text"] for r in rs if r["criterion_type"] == "exclusion")
    return note, inc, exc


# ----------------------------------------------------------------------------- benchmarks
def bench_decoding(note, inc, exc, out_len, k, repeats):
    try:
        crit = parse_criteria(inc, exc)
        if not note or not crit:
            raise gr.Error("Enter (or load) a patient note and at least one criterion on the Screening tab first.")
        with _state["lock"]:
            model, tok = get_target()
            draft = get_draft()
            sents = split_free_text(note)
            ids, _ = build_prompt_ids(tok, sents, crit[0][0], crit[0][1])
            out_len, k, repeats = int(out_len), int(k), int(repeats)
            dev = model.device.type
            rows, runs = [], {}
            for name, fn in [
                ("No cache (reference)", lambda: greedy_no_cache(model, ids, out_len, stop_on_eos=False)),
                ("KV cache", lambda: greedy_kv_cache(model, ids, out_len, stop_on_eos=False)),
                ("Speculative k=%d" % k, lambda: speculative_greedy(model, draft, ids, out_len, k=k, eos_ids=()))]:
                fn()  # warm-up
                best = None
                for _ in range(repeats):
                    if dev == "cuda":
                        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    r = fn()
                    r["mem"] = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else float("nan")
                    if best is None or r["total_s"] < best["total_s"]:
                        best = r
                runs[name] = best
            ref = runs["No cache (reference)"]
            for name, r in runs.items():
                cmp = compare_paths(r["tokens"], ref["tokens"], ref["margins"])
                extra = ""
                if "stats" in r:
                    extra = f"acceptance={r['stats']['acceptance_rate']:.2f}, mean accepted/iter={r['stats']['mean_accepted_per_iter']:.2f}"
                rows.append([name, len(ids), len(r["tokens"]), f"{r['prefill_s']*1000:.1f}",
                             f"{r['total_s']:.3f}", f"{len(r['tokens'])/r['total_s']:.1f}",
                             f"{r['mem']:.0f}", "✅ identical" if cmp["equal"] else f"❌ first diff @ {cmp['first_mismatch']}",
                             extra])
            kv_t = runs["KV cache"]["total_s"]
            summary = (f"Prompt = first criterion of current input ({len(ids)} tokens), greedy, "
                       f"{repeats} timed repeat(s) after 1 warm-up, best run shown. "
                       f"KV-cache speed-up vs no-cache: **{ref['total_s']/kv_t:.2f}×**; speculative vs KV-cache: "
                       f"**{kv_t/runs['Speculative k=%d' % k]['total_s']:.2f}×**. Target on {_state['info']['target_device']}, "
                       f"draft on {_state['info']['draft_device']}. EOS stopping is disabled so all modes produce exactly "
                       f"{out_len} tokens.")
            return summary, rows
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"{type(e).__name__}: {e}")


def bench_attention_ui(seq_len, heads, head_dim, block, causal):
    try:
        L, H, D, B = int(seq_len), int(heads), int(head_dim), int(block)
        if not (1 <= L <= 4096 and 1 <= H <= 32 and D in (16, 32, 64, 128) and 1 <= B <= 1024):
            raise gr.Error("Invalid shape: 1≤L≤4096, 1≤H≤32, D∈{16,32,64,128}, 1≤block≤1024")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator().manual_seed(0)
        q, k, v = (torch.randn(1, H, L, D, generator=g).to(dev) for _ in range(3))
        out = []
        ref = reference_attention(q, k, v, causal)
        for name, fn in [("reference (naive)", lambda: reference_attention(q, k, v, causal)),
                         (f"tiled online-softmax (block {B})", lambda: tiled_attention(q, k, v, causal, B, B)),
                         ("torch SDPA (fused backend)", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal))]:
            fn()
            if dev == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            o = fn()
            if dev == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000
            mem = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else float("nan")
            err = (o.float() - ref.float()).abs()
            out.append([name, f"{ms:.2f}", f"{mem:.1f}", f"{err.max().item():.2e}", f"{err.mean().item():.2e}"])
        return out
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {e}")


def saved_results():
    md = []
    for split in ["val", "test"]:
        for tag in ["base", "finetuned"]:
            p = os.path.join(RESULTS_DIR, f"eval_{split}_{tag}.json")
            if os.path.exists(p):
                r = json.load(open(p))
                o = r["overall"]
                md.append(f"| {split} | {tag} | {o['n']} | {o['accuracy']:.3f} | {o['macro_f1']:.3f} | "
                          f"{r['inclusion']['macro_f1']:.3f} | {r['exclusion']['macro_f1']:.3f} | "
                          f"{o['evidence']['micro_f1']:.3f} | {r['parse_errors']} |")
    if not md:
        return "No saved evaluation results found in `results/`."
    return ("| split | model | n | accuracy | macro-F1 | incl. macro-F1 | excl. macro-F1 | evidence micro-F1 | parse errors |\n"
            "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(md))


# ----------------------------------------------------------------------------- UI
with gr.Blocks(title="Clinical Trial Criterion Screener") as demo:
    gr.Markdown("# Clinical-Trial Criterion Screener\nFine-tuned Qwen2.5-1.5B-Instruct (LoRA) on the "
                "TrialGPT Criterion Annotations dataset. " + DISCLAIMER)
    with gr.Tab("Screening"):
        with gr.Row():
            with gr.Column():
                note = gr.Textbox(label="Patient record (free text, or '0. sentence' numbered lines)", lines=12)
                with gr.Row():
                    ex_idx = gr.Number(value=0, label="Validation example #", precision=0, scale=1)
                    ex_btn = gr.Button("Load validation example", scale=2)
            with gr.Column():
                inc = gr.Textbox(label="Inclusion criteria (one per line)", lines=6)
                exc = gr.Textbox(label="Exclusion criteria (one per line)", lines=6)
                mode = gr.Radio(["KV cache (fast)", "No cache (reference)", "Speculative (draft+target)"],
                                value="KV cache (fast)", label="Decoding mode")
                run = gr.Button("Run screening", variant="primary")
        status = gr.Markdown()
        table = gr.Dataframe(headers=["#", "type", "criterion", "prediction", "evidence IDs", "latency (s)"],
                             label="Criterion-level results", wrap=True)
        files = gr.File(label="Download report (HTML + JSON)", file_count="multiple")
        report = gr.HTML()
        ex_btn.click(load_example, ex_idx, [note, inc, exc])
        run.click(screen, [note, inc, exc, mode], [status, table, report, files])
    with gr.Tab("Decoding benchmark (live)"):
        gr.Markdown("Runs the hand-written **no-cache**, **KV-cache** and **speculative** decoders live on the "
                    "first criterion of the Screening tab and checks token equality against the reference.")
        with gr.Row():
            out_len = gr.Slider(8, 128, value=32, step=8, label="output tokens")
            kdraft = gr.Slider(1, 8, value=4, step=1, label="draft length k")
            reps = gr.Slider(1, 5, value=2, step=1, label="timed repeats")
        bbtn = gr.Button("Run decoding benchmark", variant="primary")
        bsum = gr.Markdown()
        btab = gr.Dataframe(headers=["mode", "input tok", "output tok", "prefill ms", "total s", "tok/s",
                                     "peak MB", "tokens vs reference", "speculative stats"], wrap=True)
        bbtn.click(bench_decoding, [note, inc, exc, out_len, kdraft, reps], [bsum, btab])
    with gr.Tab("Attention benchmark (live)"):
        with gr.Row():
            sl = gr.Slider(16, 4096, value=512, step=16, label="sequence length")
            hh = gr.Slider(1, 32, value=8, step=1, label="heads")
            hd = gr.Dropdown([16, 32, 64, 128], value=64, label="head dim")
            bl = gr.Slider(16, 512, value=128, step=16, label="tile size")
            cz = gr.Checkbox(value=True, label="causal")
        abtn = gr.Button("Run attention check", variant="primary")
        atab = gr.Dataframe(headers=["implementation", "ms", "peak MB", "max |err| vs ref", "mean |err|"])
        abtn.click(bench_attention_ui, [sl, hh, hd, bl, cz], atab)
    with gr.Tab("Saved evaluation results"):
        res_md = gr.Markdown()
        gr.Button("Refresh").click(saved_results, None, res_md)
        demo.load(saved_results, None, res_md)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)), inbrowser=True)
