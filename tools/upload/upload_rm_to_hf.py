#!/usr/bin/env python3
"""
Upload dimension-specific reward models to Hugging Face Hub.

Usage Examples:
    # Upload single dimension (default: standard type)
    python upload_rm_to_hf.py --dimension helpful
    python upload_rm_to_hf.py --dimension safe
    
    # Upload single dimension with specific type
    python upload_rm_to_hf.py --dimension helpful --type qr
    python upload_rm_to_hf.py --dimension helpful --type standard
    
    # Upload all dimensions (default: standard type)
    python upload_rm_to_hf.py --all
    
    # Upload all QR models
    python upload_rm_to_hf.py --all --type qr
    
    # Upload all standard models
    python upload_rm_to_hf.py --all --type standard
    
    # Upload both QR and standard models
    python upload_rm_to_hf.py --all --type both
    
    # Use custom username
    python upload_rm_to_hf.py --all --username your_username
    
    # Use custom token (or set HF_TOKEN environment variable)
    python upload_rm_to_hf.py --all --token your_token_here
    export HF_TOKEN='your_token_here' && python upload_rm_to_hf.py --all

Available Dimensions:
    PKU-SafeRLHF:
        - helpful, safe
    HelpSteer:
        - helpfulness, coherence, complexity, correctness, verbosity
    Hummer:
        - accuracy, conciseness, depth, empathy, specificity, tone
    Combined:
        - all

Model Types:
    - standard: Standard reward models (output/rm_{dimension}_PKU-Alignment/...)
    - qr: Quantile regression models (output/rm_qr_{dimension}_PKU-Alignment/...)
    - both: Upload both types
"""

import argparse
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


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
    
    # Check for pytorch_model.bin (file, not directory)
    pytorch_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(pytorch_path):
        return True, "pytorch_model.bin"
    
    return False, None


def upload_reward_model(dimension: str, model_type: str = "standard", hf_username: str = "imminsik", token: str = None):
    """
    Upload a dimension-specific reward model to Hugging Face Hub.
    
    Args:
        dimension: The dimension name (e.g., 'helpful', 'safe', 'accuracy')
        model_type: Model type - 'qr' for quantile regression or 'standard' for standard RM
        hf_username: Hugging Face username
        token: Hugging Face API token (if None, uses HF_TOKEN env variable)
    """
    # Determine model directory - try both path formats
    if model_type == "qr":
        candidates = [
            f"output/rm_qr_{dimension}_PKU-Alignment/alpaca-8b-reproduced-llama-3",
            f"output/rm_qr_{dimension}",
        ]
    else:  # standard RM
        candidates = [
            f"output/rm_{dimension}_PKU-Alignment/alpaca-8b-reproduced-llama-3",
            f"output/rm_{dimension}",
        ]
    
    # Find first existing directory with model files
    model_dir = None
    weight_format = None
    for candidate in candidates:
        if os.path.exists(candidate):
            has_weights, fmt = has_model_weights(candidate)
            if has_weights:
                model_dir = candidate
                weight_format = fmt
                break
    
    if model_dir is None:
        print(f"❌ No model directory found. Tried: {candidates}")
        return False
    
    # Check required files
    required_files = ["config.json", "tokenizer.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    
    if not has_weights:
        missing_files.append("model weights (safetensors or pytorch_model.bin)")
    
    if missing_files:
        print(f"❌ Missing required files in {model_dir}: {missing_files}")
        return False
    
    # Repository name
    if model_type == "qr":
        repo_name = f"safe-rlhf-qr-{dimension}-alpaca-8b-reproduced-llama-3"
    else:
        repo_name = f"safe-rlhf-rm-{dimension}-alpaca-8b-reproduced-llama-3"
    repo_id = f"{hf_username}/{repo_name}"
    
    print(f"\n📦 Uploading {dimension} reward model ({model_type})...")
    print(f"   Local path: {model_dir}")
    print(f"   Repo ID: {repo_id}")
    print(f"   Weight format: {weight_format}")
    
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
            commit_message=f"Upload {dimension} {'quantile regression ' if model_type == 'qr' else ''}reward model",
            ignore_patterns=[
                "wandb/*",           # wandb 로그
                "*.log",             # 로그 파일
                "script.sh",         # 스크립트
                "environ.txt",       # 환경 정보
                "arguments.pkl",     # 인자 pickle
                "arguments.json",    # 인자 json
                "*.backup",          # 백업 파일
                "global_step*/*",    # DeepSpeed 체크포인트 폴더
                "*optim_states.pt",  # 옵티마이저 상태 (~90GB each!)
                "latest",            # DeepSpeed latest 심볼릭 링크
                "zero_to_fp32.py",   # 변환 스크립트
            ]
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
    parser = argparse.ArgumentParser(description="Upload reward models to Hugging Face Hub")
    parser.add_argument(
        "--dimension",
        type=str,
        help="Dimension to upload (helpfulness, coherence, complexity, correctness, verbosity, qr)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["qr", "standard", "both"],
        default="standard",
        help="Model type: 'qr' for quantile regression, 'standard' for standard RM, 'both' for both types (default: standard)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload all available dimension models"
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
    
    # Define all dimensions
    # PKU-SafeRLHF dimensions (helpful/safe)
    pku_dimensions = ["helpful", "safe"]
    
    # HelpSteer dimensions
    helpsteer_dimensions = ["helpfulness", "coherence", "complexity", "correctness", "verbosity"]
    
    # Hummer dimensions
    hummer_dimensions = ["accuracy", "conciseness", "depth", "empathy", "specificity", "tone"]
    
    # Combined: all available dimensions
    all_dimensions = pku_dimensions + helpsteer_dimensions + hummer_dimensions + ["all"]
    
    standard_dimensions = all_dimensions
    qr_dimensions = all_dimensions
    
    # Determine which dimensions and types to upload
    if args.all:
        # Upload all available models
        tasks = []
        if args.type in ["qr", "both"]:
            tasks.extend([(dim, "qr") for dim in qr_dimensions])
        if args.type in ["standard", "both"]:
            tasks.extend([(dim, "standard") for dim in standard_dimensions])
    elif args.dimension:
        # Validate dimension
        if args.dimension == "qr" and args.type == "standard":
            print("❌ Error: 'qr' dimension only exists for quantile regression models")
            return
        
        valid_dims = qr_dimensions if args.type in ["qr", "both"] else standard_dimensions
        if args.dimension not in valid_dims:
            print(f"❌ Invalid dimension: {args.dimension}")
            print(f"   Valid dimensions: {', '.join(valid_dims)}")
            return
        
        # Create task list
        tasks = []
        if args.type in ["qr", "both"]:
            tasks.append((args.dimension, "qr"))
        if args.type in ["standard", "both"] and args.dimension != "qr":
            tasks.append((args.dimension, "standard"))
    else:
        print("❌ Error: Please specify --dimension or --all")
        parser.print_help()
        return
    
    # Upload each dimension
    print(f"\n🚀 Starting upload to Hugging Face (username: {args.username})")
    print("=" * 70)
    
    results = {}
    for dim, model_type in tasks:
        key = f"{dim} ({model_type})"
        success = upload_reward_model(dim, model_type, args.username, token)
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
