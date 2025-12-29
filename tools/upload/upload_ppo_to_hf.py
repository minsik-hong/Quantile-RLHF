#!/usr/bin/env python3
"""
Upload PPO-Panacea models to Hugging Face Hub.

Usage Examples:
    # Upload single PPO model (default: HH dataset, standard type)
    python upload_ppo_to_hf.py --type standard --dataset HH
    python upload_ppo_to_hf.py --type qr --dataset HH
    
    # Upload both model types for a dataset
    python upload_ppo_to_hf.py --type both --dataset HH
    
    # Upload all datasets for a model type
    python upload_ppo_to_hf.py --type standard --dataset all
    
    # Upload everything
    python upload_ppo_to_hf.py --type both --dataset all
    
    # Use custom username
    python upload_ppo_to_hf.py --type qr --username your_username
    
    # Use custom token (or set HF_TOKEN environment variable)
    python upload_ppo_to_hf.py --type standard --token your_token_here
    export HF_TOKEN='your_token_here' && python upload_ppo_to_hf.py --type both

Available Models:
    - HH: PKU-SafeRLHF dataset (output/ppo-panacea-HH-*)
    - helpsteer: HelpSteer dataset (output/ppo-panacea-*)  
    - hummer: Hummer dataset (output/ppo-panacea-hummer-*)

Model Types:
    - standard: Standard reward model based PPO
    - qr: Quantile regression reward model based PPO
    - both: Upload both types
"""

import argparse
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# 프로젝트 루트 디렉토리 (이 스크립트 기준 ../../)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def has_model_weights(model_dir: str) -> tuple:
    """
    Check if model directory has valid model weights.
    
    Returns:
        (has_weights: bool, weight_format: str or None)
    """
    import glob
    
    # Check for sharded safetensors (model-00001-of-*.safetensors)
    sharded_safetensors = glob.glob(os.path.join(model_dir, "model-*-of-*.safetensors"))
    if sharded_safetensors:
        return True, f"sharded safetensors ({len(sharded_safetensors)} files)"
    
    # Check for single safetensors
    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        return True, "model.safetensors"
    
    # Check for pytorch_model.bin (file)
    pytorch_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(pytorch_path):
        return True, "pytorch_model.bin"
    
    return False, None


def upload_ppo_model(model_type: str = "standard", dataset: str = "HH", hf_username: str = "imminsik", token: str = None):
    """
    Upload PPO-Panacea model to Hugging Face Hub.
    
    Args:
        model_type: Model type - 'qr' for quantile regression or 'standard' for standard
        dataset: Dataset type - 'HH' (PKU-SafeRLHF), 'helpsteer', or 'hummer'
        hf_username: Hugging Face username
        token: Hugging Face API token (if None, uses HF_TOKEN env variable)
    """
    # Determine model directory based on dataset and type (프로젝트 루트 기준)
    if dataset == "HH":
        if model_type == "qr":
            model_dir = str(PROJECT_ROOT / "output/ppo-panacea-HH-qr")
            repo_name = "ppo-panacea-HH-qr"
        else:
            model_dir = str(PROJECT_ROOT / "output/ppo-panacea-HH-standard")
            repo_name = "ppo-panacea-HH-standard"
    elif dataset == "hummer":
        if model_type == "qr":
            model_dir = str(PROJECT_ROOT / "output/ppo-panacea-hummer-qr")
            repo_name = "ppo-panacea-hummer-qr"
        else:
            model_dir = str(PROJECT_ROOT / "output/ppo-panacea-hummer-standard")
            repo_name = "ppo-panacea-hummer-standard"
    else:  # helpsteer or default
        if model_type == "qr":
            model_dir = str(PROJECT_ROOT / "output/ppo-panacea-qr")
            repo_name = "ppo-panacea-qr"
        else:
            model_dir = str(PROJECT_ROOT / "output/ppo-panacea-standard")
            repo_name = "ppo-panacea-standard"
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return False
    
    # Check required files
    required_files = ["config.json", "tokenizer.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    
    if missing_files:
        print(f"❌ Missing required files in {model_dir}: {missing_files}")
        return False
    
    # Check for model weights
    has_weights, weight_format = has_model_weights(model_dir)
    
    if not has_weights:
        print(f"❌ No model weights found in {model_dir}")
        print(f"   Expected: pytorch_model.bin or model*.safetensors")
        return False
    
    repo_id = f"{hf_username}/{repo_name}"
    
    print(f"\n📦 Uploading PPO-Panacea {model_type}...")
    print(f"   Local path: {model_dir}")
    print(f"   Repo ID: {repo_id}")
    print(f"   Weight format: {weight_format}")
    
    # Check panacea_config.json
    panacea_config_path = os.path.join(model_dir, "panacea_config.json")
    if os.path.exists(panacea_config_path):
        print(f"   ✅ Panacea config found")
    else:
        print(f"   ⚠️  Warning: panacea_config.json not found")
    
    # .gitignore 파일 임시 제거 (업로드 방해 방지)
    gitignore_path = os.path.join(model_dir, ".gitignore")
    gitignore_backup_path = None
    gitignore_removed = False
    
    if os.path.exists(gitignore_path):
        # .gitignore 파일을 임시로 백업하고 제거
        gitignore_backup_path = gitignore_path + ".backup"
        try:
            shutil.move(gitignore_path, gitignore_backup_path)
            gitignore_removed = True
            print(f"   ⚠️  Temporarily removed .gitignore for upload")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not remove .gitignore: {e}")
    
    try:
        # Initialize API
        api = HfApi(token=token)
        
        # Create repository (if doesn't exist)
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                token=token
            )
            print(f"✅ Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"⚠️  Repository creation warning: {e}")
        
        # Upload all files in the directory
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload PPO-Panacea {model_type} model",
            ignore_patterns=["wandb/*", "wandb/**", "logs/*", "logs/**", "*.log", "script.sh", "environ.txt", "arguments.pkl", "*.backup", "300", "global_step*", "latest", "zero_to_fp32.py"]
        )
        
        print(f"✅ Successfully uploaded to: https://huggingface.co/{repo_id}")
        success = True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        success = False
    
    finally:
        # .gitignore 파일 복원
        if gitignore_removed and gitignore_backup_path and os.path.exists(gitignore_backup_path):
            try:
                shutil.move(gitignore_backup_path, gitignore_path)
                print(f"   ✅ Restored .gitignore file")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not restore .gitignore: {e}")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Upload PPO-Panacea models to Hugging Face Hub")
    parser.add_argument(
        "--type",
        type=str,
        choices=["qr", "standard", "both"],
        default="standard",
        help="Model type: 'qr' for quantile regression, 'standard' for standard, 'both' for both types (default: standard)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["HH", "helpsteer", "hummer", "all"],
        default="HH",
        help="Dataset type: 'HH' (PKU-SafeRLHF), 'helpsteer', 'hummer', or 'all' (default: HH)"
    )
    parser.add_argument(
        "--username",
        type=str,
        default="imminsik",
        help="Hugging Face username (default: imminsik)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (default: uses HF_TOKEN env variable)"
    )
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Error: Hugging Face token not provided.")
        print("   Set HF_TOKEN environment variable or use --token argument.")
        print("   Example: export HF_TOKEN='your_token_here'")
        return
    
    # Determine which models to upload
    if args.type == "both":
        types = ["standard", "qr"]
    else:
        types = [args.type]
    
    if args.dataset == "all":
        datasets = ["HH", "helpsteer", "hummer"]
    else:
        datasets = [args.dataset]
    
    # Upload each model
    print(f"\n🚀 Starting PPO-Panacea model upload to Hugging Face (username: {args.username})")
    print("=" * 70)
    
    results = {}
    for dataset in datasets:
        for model_type in types:
            key = f"ppo-panacea-{dataset}-{model_type}"
            success = upload_ppo_model(model_type, dataset, args.username, token)
            results[key] = success
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Upload Summary:")
    for key, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {key:30s}: {status}")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nTotal: {successful}/{total} models uploaded successfully")


if __name__ == "__main__":
    main()
