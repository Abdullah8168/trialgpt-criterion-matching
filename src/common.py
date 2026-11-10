"""Shared utilities: label sets, prompt construction, output parsing, model loading.

The SAME prompt template is used for the zero-shot (before fine-tuning) model,
the fine-tuned model, training, and the application, so before/after comparisons
are on identical inputs.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")

TARGET_MODEL = os.environ.get("TARGET_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Expert label sets per criterion type (kept separate; never collapsed to yes/no).
LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
ALL_LABELS = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]

# Columns that hold previous model (GPT-4) outputs. They are dropped at load time
# and must never be used as input features or evaluation targets.
FORBIDDEN_COLUMNS = ["gpt4_explanation", "explanation_correctness",
                     "gpt4_sentences", "gpt4_eligibility"]

MAX_PROMPT_TOKENS = 1024  # documented truncation budget for the prompt

SYSTEM_PROMPT = (
    "You are a clinical-trial eligibility screening assistant. You are given a patient "
    "note whose sentences are numbered, and ONE eligibility criterion of a clinical trial. "
    "Decide how the patient relates to the criterion using ONLY the note. Absence of "
    "information is NOT evidence: if the note does not address the criterion, answer "
    "'not enough information'. Use 'not applicable' only when the criterion cannot apply to "
    "this patient (e.g. it concerns a different sex or condition). Cite the IDs of the note "
    "sentences that support your decision (empty list if none). Reply with JSON only."
)


# ----------------------------------------------------------------------------- notes
_NUM_LINE = re.compile(r"^\s*(\d+)\.\s+(.*)$")


def parse_numbered_note(note: str) -> Dict[int, str]:
    """Parse a dataset note of the form '0. sentence\n1. sentence ...'."""
    sents: Dict[int, str] = {}
    last = None
    for line in note.splitlines():
        m = _NUM_LINE.match(line)
        if m:
            last = int(m.group(1))
            sents[last] = m.group(2).strip()
        elif line.strip() and last is not None:  # continuation line
            sents[last] += " " + line.strip()
    return sents


_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_free_text(note: str) -> Dict[int, str]:
    """Split an unseen free-text clinical note into numbered sentences (app input).

    If the note is already numbered ('0. ...'), the original IDs are preserved."""
    note = note.strip()
    numbered = parse_numbered_note(note)
    if len(numbered) >= 2:
        return numbered
    pieces: List[str] = []
    for para in re.split(r"\n+", note):
        para = para.strip()
        if para:
            pieces.extend(p.strip() for p in _SPLIT.split(para) if p.strip())
    return {i: s for i, s in enumerate(pieces)}


def render_note(sents: Dict[int, str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in sorted(sents.items()))


# ----------------------------------------------------------------------------- prompt
def build_user_message(sents: Dict[int, str], criterion_type: str, criterion_text: str) -> str:
    labels = " | ".join(LABELS[criterion_type])
    return (
        f"Patient note:\n{render_note(sents)}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text.strip()}\n\n"
        f"Allowed labels: {labels}\n"
        'Answer format: {"evidence": [sentence IDs], "label": "<one allowed label>"}'
    )


def build_messages(sents, criterion_type, criterion_text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(sents, criterion_type, criterion_text)},
    ]


def target_json(label: str, evidence: List[int]) -> str:
    return json.dumps({"evidence": sorted(set(int(x) for x in evidence)), "label": label})


def build_prompt_ids(tokenizer, sents, criterion_type, criterion_text,
                     max_tokens: int = MAX_PROMPT_TOKENS) -> Tuple[List[int], int]:
    """Tokenize the chat prompt. If it exceeds the budget, drop trailing note sentences
    (documented truncation). Returns (ids, number_of_dropped_sentences)."""
    sents = dict(sents)
    dropped = 0
    while True:
        text = tokenizer.apply_chat_template(
            build_messages(sents, criterion_type, criterion_text),
            tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens or len(sents) <= 1:
            return ids, dropped
        sents.pop(max(sents))
        dropped += 1


# ----------------------------------------------------------------------------- parsing
def parse_output(text: str, criterion_type: str, valid_ids) -> Dict:
    """Robustly parse model output. Invalid labels -> 'not enough information'
    (flagged as parse_error) — never silently mapped to an eligible/ineligible label."""
    out = {"label": None, "evidence": [], "parse_error": False, "raw": text}
    m = re.search(r"\{.*\}", text, re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    allowed = LABELS[criterion_type]
    if isinstance(obj, dict):
        lab = str(obj.get("label", "")).strip().lower()
        ev = obj.get("evidence", [])
        if isinstance(ev, list):
            out["evidence"] = sorted({int(x) for x in ev
                                      if str(x).strip().lstrip("-").isdigit() and int(x) in valid_ids})
    else:
        lab = ""
        ml = re.search(r'"label"\s*:\s*"([^"]+)"', text)
        if ml:
            lab = ml.group(1).strip().lower()
    if lab not in allowed:  # try longest substring match (e.g. 'not included' before 'included')
        for cand in sorted(allowed, key=len, reverse=True):
            if cand in lab:
                lab = cand
                break
    if lab in allowed:
        out["label"] = lab
    else:
        out["label"] = "not enough information"
        out["parse_error"] = True
    return out


def explain(label: str, criterion_type: str, evidence: List[int], sents: Dict[int, str]) -> str:
    """Deterministic, evidence-grounded explanation built ONLY from cited sentences.
    (The dataset's GPT-4 explanations are never used for training or inference.)"""
    if evidence:
        cited = "; ".join(f"[{i}] \"{sents[i]}\"" for i in evidence if i in sents)
        return f"Judged '{label}' for this {criterion_type} criterion based on {cited}."
    if label == "not enough information":
        return ("No sentence in the patient record addresses this criterion; the outcome "
                "is unknown and requires additional information (absence of evidence is not "
                "treated as proof of eligibility or ineligibility).")
    return (f"Judged '{label}' without a specific cited sentence; this decision is weakly "
            f"supported and should be verified manually.")


# ----------------------------------------------------------------------------- models
def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name_or_path: str, adapter: Optional[str] = None, device: Optional[str] = None,
               dtype=None):
    """Load a causal LM (optionally with a LoRA adapter merged in) and its tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or get_device()
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype,
                                                 attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(device).eval()
    return model, tok


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
