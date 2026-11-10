"""Task 1 — LoRA supervised fine-tuning (causal-LM loss on the answer tokens only).

Usage (Kaggle T4 / Colab T4):
  python src/train_lora.py                                   # target Qwen2.5-1.5B-Instruct
  python src/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --out outputs/draft_adapter
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_DIR, OUTPUTS_DIR, RESULTS_DIR, TARGET_MODEL, build_prompt_ids, read_jsonl, write_json


def encode(tok, rec):
    sents = {int(k): v for k, v in rec["sentences"].items()}
    p_ids, dropped = build_prompt_ids(tok, sents, rec["criterion_type"], rec["criterion_text"])
    # answer + <|im_end|> so the model learns to stop
    a_ids = tok(rec["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids, "dropped": dropped}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), n), self.pad_id)
        lab = torch.full((len(feats), n), -100)
        att = torch.zeros((len(feats), n), dtype=torch.long)
        for i, f in enumerate(feats):
            L = len(f["input_ids"])
            ids[i, :L] = torch.tensor(f["input_ids"])
            lab[i, :L] = torch.tensor(f["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TARGET_MODEL)
    ap.add_argument("--out", default=os.path.join(OUTPUTS_DIR, "adapter"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=-1, help="smoke tests only")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (less memory, slower)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(args.model)
    train = [encode(tok, r) for r in read_jsonl(os.path.join(DATA_DIR, "processed", "train.jsonl"))]
    val = [encode(tok, r) for r in read_jsonl(os.path.join(DATA_DIR, "processed", "val.jsonl"))]
    print(f"train={len(train)} val={len(val)} max_len={max(len(x['input_ids']) for x in train)} "
          f"truncated={sum(x['dropped'] > 0 for x in train + val)}")

    use_cuda = torch.cuda.is_available()
    # native bf16 only on Ampere+ (T4/GTX16xx = Turing emulate it slowly) -> fp16 there
    bf16 = use_cuda and torch.cuda.get_device_capability(0)[0] >= 8
    # Frozen base weights in half precision (fp16 on T4, which has no bf16) to fit 16 GB and
    # run fast; PEFT up-casts the trainable LoRA matrices to fp32 (autocast_adapter_dtype=True),
    # so fp16 AMP + GradScaler only ever unscale fp32 gradients. (A first attempt with an fp32
    # base and batch 4 ran out of memory on the T4: the 4x649x151936 logits alone are ~1.6 GB.)
    half = torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=half, attn_implementation="sdpa")
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg, autocast_adapter_dtype=True)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    tag = os.path.basename(os.path.normpath(args.out))
    targs = TrainingArguments(
        output_dir=os.path.join(OUTPUTS_DIR, f"ckpt_{tag}"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz, per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0,
        logging_steps=5, eval_strategy="steps", eval_steps=15, save_strategy="steps",
        save_steps=15, max_steps=args.max_steps, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        fp16=use_cuda and not bf16, bf16=bf16, report_to=[], seed=args.seed,
        remove_unused_columns=False, dataloader_num_workers=0)

    def strip(ds):
        return [{"input_ids": x["input_ids"], "labels": x["labels"]} for x in ds]

    trainer = Trainer(model=model, args=targs, train_dataset=strip(train), eval_dataset=strip(val),
                      data_collator=Collator(tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    trainer.model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    log = {"model": args.model, "adapter_dir": args.out, "hyperparameters": vars(args),
           "trainable_params": trainable, "total_params": total,
           "train_examples": len(train), "val_examples": len(val),
           "train_time_s": train_time, "best_checkpoint": trainer.state.best_model_checkpoint,
           "best_eval_loss": trainer.state.best_metric,
           "gpu": torch.cuda.get_device_name(0) if use_cuda else "cpu",
           "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if use_cuda else None,
           "precision": "bf16" if bf16 else ("fp16-amp/fp32-master" if use_cuda else "fp32"),
           "log_history": trainer.state.log_history}
    write_json(log, os.path.join(RESULTS_DIR, f"train_log_{tag}.json"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in trainer.state.log_history if "eval_loss" in h]
    plt.figure(figsize=(5, 3.2))
    plt.plot(*zip(*tr), label="train loss")
    plt.plot(*zip(*ev), "o-", label="validation loss")
    plt.xlabel("optimizer step"); plt.ylabel("cross-entropy (answer tokens)")
    plt.title(f"LoRA fine-tuning: {args.model.split('/')[-1]}"); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, "figures", f"training_curve_{tag}.png"), dpi=200)
    print(json.dumps({k: v for k, v in log.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
