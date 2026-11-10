"""Criterion-level evaluation (Task 1 before/after, Task 2 metrics).

Decoding: deterministic greedy, our explicit KV-cached decoder (kv_cache.py),
max_new_tokens=48, stop at <|im_end|>. Same prompt for base and fine-tuned models.

Usage:
  python src/evaluate.py --split val  --tag base
  python src/evaluate.py --split val  --tag finetuned --adapter outputs/adapter
  python src/evaluate.py --split test --tag finetuned --adapter outputs/adapter   # FINAL only
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (DATA_DIR, RESULTS_DIR, TARGET_MODEL, build_prompt_ids, explain, load_model,
                    parse_output, read_jsonl, write_json)
from kv_cache import greedy_kv_cache
from metrics import full_report


def predict_record(model, tok, rec, max_new_tokens=48):
    sents = {int(k): v for k, v in rec["sentences"].items()}
    ids, dropped = build_prompt_ids(tok, sents, rec["criterion_type"], rec["criterion_text"])
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    g = greedy_kv_cache(model, ids, max_new_tokens, eos_ids=eos)
    text = tok.decode(g["tokens"], skip_special_tokens=True)
    p = parse_output(text, rec["criterion_type"], set(sents))
    p["evidence_text"] = {i: sents[i] for i in p["evidence"]}
    p["explanation"] = explain(p["label"], rec["criterion_type"], p["evidence"], sents)
    p["truncated_sentences"] = dropped
    p["n_new_tokens"] = len(g["tokens"])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=TARGET_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    recs = read_jsonl(os.path.join(DATA_DIR, "processed", f"{args.split}.jsonl"))
    if args.limit:
        recs = recs[: args.limit]
    model, tok = load_model(args.model, args.adapter)
    preds = []
    t0 = time.time()
    for i, r in enumerate(recs):
        preds.append(predict_record(model, tok, r))
        if i % 25 == 0:
            print(f"[{i}/{len(recs)}] gold={r['label']!r} pred={preds[-1]['label']!r} "
                  f"raw={preds[-1]['raw'][:80]!r}", flush=True)
    rep = full_report(recs, preds)
    rep.update({"split": args.split, "tag": args.tag, "model": args.model, "adapter": args.adapter,
                "decoding": "greedy, explicit KV cache, max_new_tokens=48, stop=<|im_end|>, fp16",
                "wall_time_s": time.time() - t0})
    write_json(rep, os.path.join(RESULTS_DIR, f"eval_{args.split}_{args.tag}.json"))
    with open(os.path.join(RESULTS_DIR, f"preds_{args.split}_{args.tag}.jsonl"), "w", encoding="utf-8") as f:
        for r, p in zip(recs, preds):
            f.write(json.dumps({"annotation_id": r["annotation_id"], "patient_id": r["patient_id"],
                                "criterion_type": r["criterion_type"],
                                "criterion_text": r["criterion_text"], "gold_label": r["label"],
                                "gold_evidence": r["evidence"], **p}) + "\n")
    o = rep["overall"]
    print(f"{args.split}/{args.tag}: acc={o['accuracy']:.3f} macroF1={o['macro_f1']:.3f} "
          f"evid microF1={o['evidence']['micro_f1']:.3f} parse_errors={rep['parse_errors']}")


if __name__ == "__main__":
    main()
