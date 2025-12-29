"""
자동화 스크립트: 모든 분석 스크립트를 데이터셋별로 실행

Usage:
    python run_all_analysis.py --dataset pku
    python run_all_analysis.py --dataset helpsteer
    python run_all_analysis.py --dataset hummer
    python run_all_analysis.py --dataset hummer-all
    python run_all_analysis.py --all  # 모든 데이터셋
    
    # 최신 로그 폴더 자동 탐색
    python run_all_analysis.py --dataset pku --auto-latest
"""

import os
import sys
import subprocess
import argparse
import pandas as pd

from config import get_dataset_config, AVAILABLE_DATASETS, get_output_dir


# 분석 스크립트 목록
ANALYSIS_SCRIPTS = [
    "comprehensive_statistical_analysis.py",
    "comprehensive_roc_analysis.py",
    "hard_case_analysis.py",
    "compare_qr_vs_standard_kde.py",
    "compare_qr_vs_standard_histogram.py",
]

# Conflict analysis는 PKU (helpful/safe)에서만 의미 있음
CONFLICT_SCRIPTS = [
    "conflict_analysis.py",
]


def run_quantile_to_expectation(config):
    """quantile_to_expectation: QR 데이터의 기대값 계산"""
    print("="*80)
    print("Step 1: Computing QR expectation values")
    print("="*80)
    
    qr_paths = config.get("qr_paths", {})
    
    for dim, qr_path in qr_paths.items():
        if not os.path.exists(qr_path):
            print(f"  [SKIP] {dim}: {qr_path} not found")
            continue
        
        for fname in ["higher_quantiles.csv", "lower_quantiles.csv"]:
            fpath = os.path.join(qr_path, fname)
            if not os.path.exists(fpath):
                print(f"  [SKIP] {fpath} not found")
                continue
            
            df = pd.read_csv(fpath)
            df["q_expectation"] = df.mean(axis=1)
            
            out_name = fname.replace(".csv", "_expectation.csv")
            out_path = os.path.join(qr_path, out_name)
            df[["q_expectation"]].to_csv(out_path, index=False)
            
            print(f"  [DONE] {dim}: {out_name} (rows={len(df)})")
    
    print("\n")


def run_analysis_script(script_name, dataset, auto_latest=False):
    """분석 스크립트 실행"""
    print("="*80)
    cmd_str = f"--dataset {dataset}"
    if auto_latest:
        cmd_str += " --auto-latest"
    print(f"Running {script_name} {cmd_str}")
    print("="*80)
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"[ERROR] {script_path} not found")
        return False
    
    cmd = [sys.executable, script_path, "--dataset", dataset]
    if auto_latest:
        cmd.append("--auto-latest")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(__file__),
            capture_output=False
        )
        if result.returncode != 0:
            print(f"[ERROR] {script_name} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"[ERROR] {script_name}: {e}")
        return False
    
    print("\n")
    return True


def run_for_dataset(dataset, auto_latest=False):
    """단일 데이터셋에 대해 모든 분석 실행"""
    print("\n" + "="*80)
    print(f"Starting Analysis Pipeline for Dataset: {dataset.upper()}")
    if auto_latest:
        print("(Auto-latest mode: finding latest log folders)")
    print("="*80 + "\n")
    
    config = get_dataset_config(dataset, auto_latest=auto_latest)
    print(f"Dataset: {config['name']}")
    print(f"Dimensions: {config['dimensions']}")
    print(f"Output Dir: {get_output_dir(dataset)}")
    print("="*80 + "\n")
    
    # Step 1: QR expectation 계산
    run_quantile_to_expectation(config)
    
    # Step 2: 일반 분석 스크립트 실행
    for script in ANALYSIS_SCRIPTS:
        run_analysis_script(script, dataset, auto_latest)
    
    # Step 3: Conflict analysis (PKU only)
    if dataset == "pku":
        for script in CONFLICT_SCRIPTS:
            run_analysis_script(script, dataset, auto_latest)
    
    print("="*80)
    print(f"Analysis complete for {dataset.upper()}!")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run all analysis scripts")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=AVAILABLE_DATASETS,
        help=f"Dataset to analyze: {AVAILABLE_DATASETS}"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run analysis for all datasets"
    )
    parser.add_argument(
        "--auto-latest",
        action="store_true",
        help="Automatically find the latest log folders in eval_tables"
    )
    
    args = parser.parse_args()
    
    if args.all:
        datasets = AVAILABLE_DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Error: Please specify --dataset or --all")
        parser.print_help()
        return
    
    for dataset in datasets:
        run_for_dataset(dataset, auto_latest=args.auto_latest)
    
    print("\n" + "="*80)
    print("All analysis pipelines completed!")
    print("="*80)


if __name__ == "__main__":
    main()
