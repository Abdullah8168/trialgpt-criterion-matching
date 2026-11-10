# Clinical-Trial Criterion Matching with a Fine-Tuned Transformer

Educational research prototype for **Generative AI — Assignment 2**. It matches a patient description
against individual inclusion and exclusion criteria of a clinical trial, using **Qwen2.5-1.5B-Instruct**
fine-tuned with LoRA on the **TrialGPT Criterion Annotations** dataset. For each criterion it returns
a label, the IDs of the patient-note sentences cited as evidence, the evidence text and an explanation
built from those sentences. It also includes hand-written implementations of **KV-cached decoding**,
**tiled online-softmax attention** and **greedy speculative decoding**, plus a browser application.

> ⚠️ This is an educational screening prototype, **not** a clinical enrollment decision tool.

## Repository layout
| Path | Contents |
|---|---|
| `src/prepare_data.py` | Fixed 37/8/8 patient-grouped split (seed 42), removal of GPT-4 columns, leakage checks, truncation stats |
| `data/split_manifest.json` | Published split manifest (patient IDs, counts, label distribution, truncation) |
| `src/common.py` | Prompt template, output schema, parser, evidence-grounded explanation |
| `src/train_lora.py` | LoRA supervised fine-tuning (loss on answer tokens only) |
| `src/evaluate.py`, `src/metrics.py` | Criterion and evidence metrics (overall, inclusion, exclusion) |
| `src/kv_cache.py`, `src/bench_kv.py` | Explicit KV-cache decoder, no-cache reference, equality tests and benchmark |
| `src/tiled_attention.py`, `src/bench_attention.py` | Reference SDPA and tiled online-softmax attention, with validation and benchmark |
| `src/speculative.py`, `src/bench_speculative.py` | Greedy speculative decoding (draft, verify, accept, correct, cache crop) |
| `app.py`, `start_app.bat`, `start_app.sh` | Gradio browser application |
| `notebooks/kaggle_pipeline.ipynb` | End-to-end GPU pipeline (Kaggle T4) |
| `results/` | Evaluation JSONs, predictions, benchmark logs, figures |
| `report/` | IEEE LaTeX report |

## Output schema
```json
{"evidence": [3, 4], "label": "included"}
```
Labels: inclusion ∈ {included, not included, not enough information, not applicable};
exclusion ∈ {excluded, not excluded, not enough information, not applicable}.

## Reproduce
1. **GPU pipeline (Kaggle, free T4).** Upload `notebooks/kaggle_pipeline.ipynb`, set `REPO_URL`, and
   choose Accelerator GPU T4 and Internet On. Then run all cells (about 45 min). Download `artifacts.zip` and
   unzip it into the repo root. This adds `results/`, `outputs/adapter`, `outputs/draft_adapter` and `data/processed`.
2. **Local, step by step:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   python src/prepare_data.py
   python src/evaluate.py --split val --tag base
   python src/train_lora.py
   python src/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --out outputs/draft_adapter
   python src/evaluate.py --split val --tag finetuned --adapter outputs/adapter
   python src/evaluate.py --split test --tag base
   python src/evaluate.py --split test --tag finetuned --adapter outputs/adapter
   python src/bench_kv.py --adapter outputs/adapter
   python src/bench_attention.py
   python src/bench_speculative.py --adapter outputs/adapter --draft_adapter outputs/draft_adapter
   python src/make_figures.py
   ```

## Run the application
- **Windows:** double-click `start_app.bat`. **Linux/macOS:** `./start_app.sh`.
- The launcher creates the environment, downloads the adapters if they are missing, and opens <http://127.0.0.1:7860>.
- Adapter download: `python scripts/download_weights.py` (set `HF_ADAPTER_REPO`).
- Hardware: a GPU with 4 GB or more runs the target model in fp16. If GPU memory is short, the draft model is placed
  on the CPU automatically. CPU-only also works, but it is slow.

Tabs: **Screening** (note and criteria in, criterion-level results, evidence, missing information,
aggregated preliminary outcome, HTML/JSON report download), **Decoding benchmark** (live no-cache vs KV
vs speculative, with token equality), **Attention benchmark** (live reference vs tiled vs fused),
**Saved evaluation results**.

## AI-use acknowledgement
Code was written with the help of an AI coding assistant (Claude). All components were run, tested
and checked by the author. See the report appendix.
