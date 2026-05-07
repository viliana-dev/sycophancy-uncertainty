"""Upload dataset to HuggingFace Hub."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi

REPO_ID = os.environ.get("HF_REPO_ID", "<your-username>/nested-geometry-sycophancy")
TOKEN = os.environ["HF_TOKEN"]
PROJECT = Path(__file__).resolve().parent.parent

api = HfApi(token=TOKEN)

# Create repo
print(f"Creating repo {REPO_ID}...", flush=True)
api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

# Upload README
readme_path = PROJECT / "hf_README.md"
if readme_path.exists():
    print("Uploading README.md...", flush=True)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

# Upload data/generated (labeled, entropy, splits, steering)
for suffix in ["sycophancy", "sycophancy_gptoss", "sycophancy_qwen_moe"]:
    gen_dir = PROJECT / "data" / "generated" / suffix
    if not gen_dir.exists():
        print(f"SKIP {gen_dir} (not found)")
        continue
    
    for fname in ["labeled.jsonl", "answer_entropy.jsonl", "splits.json"]:
        fpath = gen_dir / fname
        if fpath.exists():
            dest = f"data/generated/{suffix}/{fname}"
            print(f"  Uploading {dest} ({fpath.stat().st_size / 1e6:.1f} MB)...", flush=True)
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=dest,
                repo_id=REPO_ID,
                repo_type="dataset",
            )

    # Steering dir
    steer_dir = gen_dir / "steering"
    if steer_dir.exists() and any(steer_dir.iterdir()):
        print(f"  Uploading {suffix}/steering/ ...", flush=True)
        api.upload_folder(
            folder_path=str(steer_dir),
            path_in_repo=f"data/generated/{suffix}/steering",
            repo_id=REPO_ID,
            repo_type="dataset",
        )

# Upload data/features (all NPZ files)
for suffix in ["sycophancy", "sycophancy_gptoss", "sycophancy_qwen_moe"]:
    feat_dir = PROJECT / "data" / "features" / suffix
    if not feat_dir.exists():
        print(f"SKIP features/{suffix} (not found)")
        continue
    
    npz_files = sorted(feat_dir.glob("*.npz"))
    print(f"\n  Uploading features/{suffix}/ ({len(npz_files)} NPZ files)...", flush=True)
    api.upload_folder(
        folder_path=str(feat_dir),
        path_in_repo=f"data/features/{suffix}",
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["*.npz"],
    )

# Upload results/*.json
results_dir = PROJECT / "results"
if results_dir.exists():
    json_files = sorted(results_dir.glob("*.json"))
    print(f"\nUploading results/ ({len(json_files)} JSON files)...", flush=True)
    api.upload_folder(
        folder_path=str(results_dir),
        path_in_repo="results",
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["*.json"],
    )

print("\nDone! Check: https://huggingface.co/datasets/" + REPO_ID)
