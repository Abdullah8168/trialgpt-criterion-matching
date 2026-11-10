"""C4 — categorize errors of the fine-tuned model on the held-out test split and pick
>= 8 distinct, representative cases (written to results/error_analysis.md and a LaTeX table).

Categories:
  UNSUPPORTED      gold = not enough information, model made a decisive call (hallucinated certainty)
  MISSED_INFO      gold decisive, model said not enough information (failed to find/use evidence)
  CONTRADICTION    model chose the opposite decisive label (e.g. included <-> not included)
  NOT_APPLICABLE   confusion involving 'not applicable'
  EVIDENCE_ONLY    label correct but cited sentences disjoint from gold evidence
  PARSE            unparseable output (defaulted to not enough information)
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS_DIR, ROOT, read_jsonl

DECISIVE = {"included", "not included", "excluded", "not excluded"}


def category(p):
    g, y = p["gold_label"], p["label"]
    if p.get("parse_error"):
        return "PARSE"
    if g == y:
        if p["gold_evidence"] and not set(p["gold_evidence"]) & set(p["evidence"]):
            return "EVIDENCE_ONLY"
        return None
    if "not applicable" in (g, y):
        return "NOT_APPLICABLE"
    if g == "not enough information":
        return "UNSUPPORTED"
    if y == "not enough information":
        return "MISSED_INFO"
    return "CONTRADICTION"


def main(split="test", tag="finetuned", n_per_cat=2):
    preds = read_jsonl(os.path.join(RESULTS_DIR, f"preds_{split}_{tag}.jsonl"))
    sents = {r["annotation_id"]: r["sentences"] for r in
             read_jsonl(os.path.join(ROOT, "data", "processed", f"{split}.jsonl"))}
    by = defaultdict(list)
    for p in preds:
        c = category(p)
        if c:
            by[c].append(p)
    counts = Counter({k: len(v) for k, v in by.items()})
    picked = []
    for c in ["UNSUPPORTED", "MISSED_INFO", "CONTRADICTION", "NOT_APPLICABLE", "EVIDENCE_ONLY", "PARSE"]:
        picked += [(c, p) for p in by.get(c, [])[:n_per_cat]]
    # top up to >= 8 cases
    i = 0
    rest = [(c, p) for c in by for p in by[c][n_per_cat:]]
    while len(picked) < 8 and i < len(rest):
        picked.append(rest[i]); i += 1

    md = [f"# Error analysis ({split}, {tag})\n", "Error counts by category: " +
          ", ".join(f"{k}={v}" for k, v in counts.most_common()) + f" (of {len(preds)} records)\n"]
    tex = []
    for k, (c, p) in enumerate(picked, 1):
        s = sents[p["annotation_id"]]
        gold_txt = "; ".join(f"[{i}] {s[str(i)]}" for i in p["gold_evidence"]) or "(none)"
        pred_txt = "; ".join(f"[{i}] {s[str(i)]}" for i in p["evidence"]) or "(none)"
        md.append(f"## E{k}. {c} — patient {p['patient_id']}, {p['criterion_type']}\n"
                  f"- **Criterion:** {p['criterion_text']}\n- **Gold:** {p['gold_label']} | evidence {gold_txt}\n"
                  f"- **Predicted:** {p['label']} | evidence {pred_txt}\n- **Raw output:** `{p['raw']}`\n")
        crit = p["criterion_text"][:70].replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
        tex.append(f"E{k} & {c.replace('_', ' ').lower()} & {p['criterion_type'][:3]}. & {crit}{'...' if len(p['criterion_text']) > 70 else ''} & "
                   f"{p['gold_label']} & {p['label']} \\\\")
    with open(os.path.join(RESULTS_DIR, "error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    json.dump(dict(counts), open(os.path.join(RESULTS_DIR, "error_counts.json"), "w"), indent=2)
    os.makedirs(os.path.join(ROOT, "report", "tables"), exist_ok=True)
    with open(os.path.join(ROOT, "report", "tables", "errors.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(counts, "picked", len(picked))


if __name__ == "__main__":
    main()
