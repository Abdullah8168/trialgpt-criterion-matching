"""Download the trained LoRA adapters (target + draft) from the Hugging Face Hub.

Set HF_ADAPTER_REPO (default below) to the repo where adapters were uploaded with
scripts/upload_weights.py. Base models are downloaded automatically by transformers.
"""
import os
from huggingface_hub import snapshot_download

REPO = os.environ.get("HF_ADAPTER_REPO", "Abdullah8168/trialgpt-qwen2.5-lora")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = snapshot_download(REPO, local_dir=os.path.join(ROOT, "outputs"))
print("Adapters downloaded to", path, "->", os.listdir(path))
