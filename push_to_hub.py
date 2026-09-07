import os
import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload trained RoboMamba + DINOv2 + ACT policy files to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Repository ID (e.g. 'username/RoboMamba_DINO_ACT_SO101')")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Local directory containing the checkpoints")
    parser.add_argument("--private", action="store_true", help="Make the Hugging Face repository private")
    args = parser.parse_args()

    api = HfApi()

    print(f"Ensuring Hugging Face repository '{args.repo_id}' exists...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    # Files required for running inference / evaluation
    files_to_upload = {
        "best_model.pt": os.path.join(args.checkpoint_dir, "best_model.pt"),
        "latest_model.pt": os.path.join(args.checkpoint_dir, "latest_model.pt"),
        "dataset_stats.pth": os.path.join(args.checkpoint_dir, "dataset_stats.pth"),
        "config.yaml": "config.yaml",
        "README.md": "README.md",
    }

    print("Uploading policy checkpoint files to Hugging Face Hub...")
    uploaded_any = False
    for path_in_repo, local_path in files_to_upload.items():
        if os.path.exists(local_path):
            print(f" -> Uploading '{local_path}' as '{path_in_repo}'...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type="model",
            )
            uploaded_any = True
        else:
            print(f" -> Skipping '{local_path}' (file does not exist)")

    if uploaded_any:
        print(f"\n Success! Your model checkpoint is live at: https://huggingface.co/{args.repo_id}")
    else:
        print("\n Error: No checkpoint files were found to upload. Please verify training ran successfully.")

if __name__ == "__main__":
    main()
