"""Upload outputs/adapter and outputs/draft_adapter to the Hugging Face Hub.
Usage: huggingface-cli login ; python scripts/upload_weights.py <hf-username>/trialgpt-qwen2.5-lora"""
import os, sys
from huggingface_hub import HfApi
repo = sys.argv[1]
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
api = HfApi()
api.create_repo(repo, exist_ok=True)
for sub in ["adapter", "draft_adapter"]:
    if os.path.isdir(os.path.join(root, sub)):
        api.upload_folder(folder_path=os.path.join(root, sub), path_in_repo=sub, repo_id=repo)
print("uploaded to https://huggingface.co/" + repo)
