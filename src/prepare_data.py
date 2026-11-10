"""Data preparation: fixed patient-grouped 37/8/8 split (seed 42), leakage checks,
GPT-4 column removal, sentence-ID parsing, tokenization/truncation statistics.

Usage: python src/prepare_data.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (DATA_DIR, FORBIDDEN_COLUMNS, LABELS, MAX_PROMPT_TOKENS, TARGET_MODEL,
                    parse_numbered_note, target_json, write_json)

RAW = os.path.join(DATA_DIR, "raw", "trialgpt_criterion.parquet")
URL = ("https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations/resolve/main/"
       "data/train-00000-of-00001.parquet")


def main():
    if not os.path.exists(RAW):
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(URL, RAW)
    df = pd.read_parquet(RAW)
    n_raw = len(df)

    # 1) Remove previous model predictions/explanations (never used as inputs or labels).
    df = df.drop(columns=[c for c in FORBIDDEN_COLUMNS if c in df.columns])
    # The dataset's own 'training' flag is ignored: the assignment prescribes its own split.
    df = df.drop(columns=["training"], errors="ignore")
    assert not any(c.startswith("gpt4") for c in df.columns)

    # 2) Prescribed split: sort unique IDs, shuffle with default_rng(42), 37/8/8.
    ids = np.array(sorted(df["patient_id"].unique()))
    assert len(ids) == 53, len(ids)
    rng = np.random.default_rng(42)
    rng.shuffle(ids)
    split_of = {}
    for i, pid in enumerate(ids):
        split_of[pid] = "train" if i < 37 else ("val" if i < 45 else "test")
    df["split"] = df["patient_id"].map(split_of)

    # 3) Leakage checks: each patient in exactly one split.
    per_patient = df.groupby("patient_id")["split"].nunique()
    assert (per_patient == 1).all(), "patient appears in more than one split"
    sets = {s: set(df.loc[df.split == s, "patient_id"]) for s in ["train", "val", "test"]}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])

    # 4) Build records with sentence IDs.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    from common import build_prompt_ids
    out = {"train": [], "val": [], "test": []}
    trunc_stats = {"max_prompt_tokens": MAX_PROMPT_TOKENS, "records_truncated": 0,
                   "records_truncated_with_gold_evidence_lost": 0, "prompt_len": []}
    bad_label = 0
    empty_criterion = 0
    for r in df.itertuples():
        if not isinstance(r.criterion_text, str) or not r.criterion_text.strip():
            empty_criterion += 1   # 1 row has a null criterion text: unusable, dropped & reported
            continue
        sents = parse_numbered_note(r.note)
        gold_ev = [int(x) for x in json.loads(r.expert_sentences)]
        assert all(e in sents for e in gold_ev), (r.annotation_id, gold_ev)
        lab = r.expert_eligibility.strip().lower()
        if lab not in LABELS[r.criterion_type]:
            bad_label += 1
            continue
        ids_, dropped = build_prompt_ids(tok, sents, r.criterion_type, r.criterion_text)
        trunc_stats["prompt_len"].append(len(ids_))
        if dropped:
            trunc_stats["records_truncated"] += 1
            kept = len(sents) - dropped
            if any(e >= kept for e in gold_ev):
                trunc_stats["records_truncated_with_gold_evidence_lost"] += 1
        out[r.split].append({
            "annotation_id": int(r.annotation_id), "patient_id": r.patient_id,
            "trial_id": r.trial_id, "trial_title": r.trial_title,
            "criterion_type": r.criterion_type, "criterion_text": r.criterion_text,
            "sentences": {str(k): v for k, v in sents.items()},
            "label": lab, "evidence": sorted(gold_ev),
            "target": target_json(lab, gold_ev),
        })
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
    for s, recs in out.items():
        with open(os.path.join(DATA_DIR, "processed", f"{s}.jsonl"), "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")

    pl = np.array(trunc_stats.pop("prompt_len"))
    trunc_stats.update({"prompt_tokens_mean": float(pl.mean()), "prompt_tokens_max": int(pl.max()),
                        "prompt_tokens_p95": float(np.percentile(pl, 95)), "tokenizer": TARGET_MODEL})

    def label_counts(s):
        from collections import Counter
        c = Counter(f"{r['criterion_type']}/{r['label']}" for r in out[s])
        return dict(sorted(c.items()))

    manifest = {
        "dataset": "ncbi/TrialGPT-Criterion-Annotations",
        "procedure": "sorted unique patient_id -> numpy.random.default_rng(42).shuffle -> "
                     "first 37 train, next 8 val, last 8 test",
        "removed_columns": FORBIDDEN_COLUMNS + ["training"],
        "n_rows_raw": n_raw, "n_rows_with_invalid_label_dropped": bad_label,
        "n_rows_with_null_criterion_text_dropped": empty_criterion,
        "shuffled_patient_order": ids.tolist(),
        "splits": {s: {"patient_ids": sorted(sets[s]), "n_patients": len(sets[s]),
                       "n_annotations": len(out[s]),
                       "n_inclusion": sum(r["criterion_type"] == "inclusion" for r in out[s]),
                       "n_exclusion": sum(r["criterion_type"] == "exclusion" for r in out[s]),
                       "n_with_gold_evidence": sum(bool(r["evidence"]) for r in out[s]),
                       "label_counts": label_counts(s)}
                   for s in ["train", "val", "test"]},
        "leakage_check": "passed: every patient_id belongs to exactly one split; pairwise "
                         "intersections of split patient sets are empty",
        "truncation": trunc_stats,
    }
    write_json(manifest, os.path.join(DATA_DIR, "split_manifest.json"))
    for s in ["train", "val", "test"]:
        m = manifest["splits"][s]
        print(f"{s}: {m['n_patients']} patients, {m['n_annotations']} annotations")
    print("truncation:", trunc_stats)


if __name__ == "__main__":
    main()
