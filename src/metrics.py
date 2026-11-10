"""Classification and evidence-retrieval metrics.

Evidence handling of empty sets (stated explicitly, as required):
* Micro P/R/F1: pool (record, sentence-id) pairs over ALL records. Records with empty gold
  and empty prediction contribute nothing; spurious citations on empty-gold records count
  as false positives.
* Macro P/R/F1: averaged over records whose GOLD evidence set is non-empty. If the
  prediction is empty for such a record, precision = recall = F1 = 0.
* Empty-gold agreement: fraction of empty-gold records for which the model also cited
  nothing (reported separately, so they are neither ignored nor rewarded silently).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from common import ALL_LABELS, LABELS


def classification_report(gold: List[str], pred: List[str], labels: List[str]) -> Dict:
    present = [l for l in labels if l in gold or l in pred]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=present, zero_division=0)
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, pred)) if gold else None,
        "macro_f1": float(f1_score(gold, pred, labels=[l for l in labels if l in gold],
                                   average="macro", zero_division=0)) if gold else None,
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, l in enumerate(present)},
        "confusion_matrix": {"labels": present,
                             "matrix": confusion_matrix(gold, pred, labels=present).tolist()},
    }


def evidence_metrics(gold_sets: List[List[int]], pred_sets: List[List[int]]) -> Dict:
    tp = fp = fn = 0
    ps, rs, fs = [], [], []
    empty_gold, empty_agree = 0, 0
    for g, p in zip(gold_sets, pred_sets):
        g, p = set(g), set(p)
        inter = len(g & p)
        tp += inter
        fp += len(p - g)
        fn += len(g - p)
        if not g:
            empty_gold += 1
            empty_agree += int(not p)
            continue
        pr = inter / len(p) if p else 0.0
        rc = inter / len(g)
        ps.append(pr); rs.append(rc)
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    mp = tp / (tp + fp) if tp + fp else 0.0
    mr = tp / (tp + fn) if tp + fn else 0.0
    return {
        "micro_precision": mp, "micro_recall": mr,
        "micro_f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
        "macro_precision": float(np.mean(ps)) if ps else None,
        "macro_recall": float(np.mean(rs)) if rs else None,
        "macro_f1": float(np.mean(fs)) if fs else None,
        "n_records_with_gold": len(ps), "n_records_empty_gold": empty_gold,
        "empty_gold_agreement": empty_agree / empty_gold if empty_gold else None,
    }


def full_report(records: List[Dict], preds: List[Dict]) -> Dict:
    rep = {}
    for name, sel in [("overall", lambda r: True),
                      ("inclusion", lambda r: r["criterion_type"] == "inclusion"),
                      ("exclusion", lambda r: r["criterion_type"] == "exclusion")]:
        idx = [i for i, r in enumerate(records) if sel(r)]
        labels = ALL_LABELS if name == "overall" else LABELS[name]
        rep[name] = classification_report([records[i]["label"] for i in idx],
                                          [preds[i]["label"] for i in idx], labels)
        rep[name]["evidence"] = evidence_metrics([records[i]["evidence"] for i in idx],
                                                 [preds[i]["evidence"] for i in idx])
    rep["parse_errors"] = sum(p.get("parse_error", False) for p in preds)
    return rep
