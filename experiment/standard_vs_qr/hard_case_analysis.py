"""
High Spread Analysis: Assessing Performance on Most Uncertain Samples

Spread (q9 - q0)가 큰 샘플 = 모델이 예측값에 대해 가장 불확실해하는 '어려운' 샘플
이 구간에서 Standard와 QR의 Ranking Accuracy 비교 및 데이터 확인

Usage:
    python hard_case_analysis.py --dataset pku
    python hard_case_analysis.py --dataset helpsteer
    python hard_case_analysis.py --dataset hummer
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
from sklearn.metrics import roc_auc_score

from config import get_dataset_config, add_dataset_argument


def get_jsonl_line(file_path, line_idx):
    """JSONL 파일에서 특정 라인 읽기"""
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    return json.loads(line)
    except Exception:
        return None
    return None


def compute_auc(h, l):
    """AUC 계산"""
    y_true = np.concatenate([np.ones(len(h)), np.zeros(len(l))])
    y_score = np.concatenate([h, l])
    return roc_auc_score(y_true, y_score)


def analyze_high_spread(dim, config, top_percent=10):
    """High Spread 분석"""
    standard_paths = config["standard_paths"]
    qr_paths = config["qr_paths"]
    dataset_paths = config.get("dataset_paths", {})
    
    std_path = standard_paths.get(dim)
    qr_path = qr_paths.get(dim)
    dataset_path = dataset_paths.get(dim)
    
    if not std_path or not qr_path:
        return None
    if not os.path.exists(std_path) or not os.path.exists(qr_path):
        print(f"[SKIP] {dim}: paths not found")
        return None
    
    try:
        std_h = pd.read_csv(os.path.join(std_path, "higher_rewards.csv"))["reward"]
        std_l = pd.read_csv(os.path.join(std_path, "lower_rewards.csv"))["reward"]
        qr_h = pd.read_csv(os.path.join(qr_path, "higher_quantiles.csv"))
        qr_l = pd.read_csv(os.path.join(qr_path, "lower_quantiles.csv"))
    except Exception as e:
        print(f"[ERROR] {dim}: {e}")
        return None
    
    # Spread 계산
    q0_col = 'q0' if 'q0' in qr_h.columns else qr_h.columns[0]
    q9_col = 'q9' if 'q9' in qr_h.columns else qr_h.columns[-1]
    
    spread_h = qr_h[q9_col] - qr_h[q0_col]
    spread_l = qr_l[q9_col] - qr_l[q0_col]
    avg_pair_spread = (spread_h + spread_l) / 2
    
    qr_h_mean = qr_h.mean(axis=1)
    qr_l_mean = qr_l.mean(axis=1)
    
    df = pd.DataFrame({
        'original_idx': range(len(std_h)),
        'std_h': std_h, 'std_l': std_l,
        'qr_h_mean': qr_h_mean, 'qr_l_mean': qr_l_mean,
        'qr_h_q0': qr_h[q0_col], 'qr_h_q9': qr_h[q9_col],
        'qr_l_q0': qr_l[q0_col], 'qr_l_q9': qr_l[q9_col],
        'pair_spread': avg_pair_spread,
        'margin_std': std_h - std_l,
        'margin_qr': qr_h_mean - qr_l_mean,
        'correct_std': (std_h > std_l),
        'correct_qr': (qr_h_mean > qr_l_mean)
    })
    
    # Overall
    overall_auc_std = compute_auc(df['std_h'], df['std_l'])
    overall_auc_qr = compute_auc(df['qr_h_mean'], df['qr_l_mean'])
    overall_acc_std = df['correct_std'].mean()
    overall_acc_qr = df['correct_qr'].mean()
    
    # High Spread Subset
    quantile_val = 1.0 - (top_percent / 100.0)
    threshold = df['pair_spread'].quantile(quantile_val)
    high_spread_df = df[df['pair_spread'] >= threshold].sort_values(by='pair_spread', ascending=False)
    
    if len(high_spread_df) > 0:
        high_auc_std = compute_auc(high_spread_df['std_h'], high_spread_df['std_l'])
        high_auc_qr = compute_auc(high_spread_df['qr_h_mean'], high_spread_df['qr_l_mean'])
        high_acc_std = high_spread_df['correct_std'].mean()
        high_acc_qr = high_spread_df['correct_qr'].mean()
    else:
        high_auc_std = high_auc_qr = high_acc_std = high_acc_qr = 0
    
    results = {
        "overall": {"acc_std": overall_acc_std, "acc_qr": overall_acc_qr, 
                   "auc_std": overall_auc_std, "auc_qr": overall_auc_qr},
        "subset": {"acc_std": high_acc_std, "acc_qr": high_acc_qr, 
                  "auc_std": high_auc_std, "auc_qr": high_auc_qr, 
                  "count": len(high_spread_df)}
    }
    
    print(f"\n" + "="*110)
    print(f" [{dim.upper()}] HIGH SPREAD ANALYSIS (Top {top_percent}%)")
    print("="*110)
    print(f"Total Pairs: {len(df)} | Overall Accuracy: Std={overall_acc_std:.4f}, QR={overall_acc_qr:.4f}")
    print(f"\n🔥 High Uncertainty Samples (Top {top_percent}% Spread, n={len(high_spread_df)})")
    print(f"Accuracy in this region: Standard={high_acc_std:.4f}, QR={high_acc_qr:.4f}")
    
    # 샘플 출력 (상위 3개)
    print(f"\n{'Line #':>7} | {'Spread':>7} | {'Std H/L':>14} | {'QR Mean H/L':>14} | {'Res (S|Q)'}")
    print("-" * 80)
    for _, row in high_spread_df.head(3).iterrows():
        line_num = int(row['original_idx']) + 1
        res = f"{'✅' if row['correct_std'] else '❌'}|{'✅' if row['correct_qr'] else '❌'}"
        std_hl = f"{row['std_h']:5.1f}/{row['std_l']:5.1f}"
        qr_hl = f"{row['qr_h_mean']:5.1f}/{row['qr_l_mean']:5.1f}"
        print(f"L{line_num:5d} | {row['pair_spread']:7.4f} | {std_hl:14} | {qr_hl:14} |   {res}")
        
        if dataset_path:
            data = get_jsonl_line(dataset_path, int(row['original_idx']))
            if data:
                prompt = data.get('prompt', 'N/A')[:100]
                print(f"   [Prompt]: {prompt}...")
                print("-" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="High Spread Analysis")
    add_dataset_argument(parser)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"High Spread Analysis - Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    config = get_dataset_config(args.dataset, auto_latest=getattr(args, "auto_latest", False))
    dimensions = config["dimensions"]
    
    PERCENTAGES = [10, 5, 1]
    summary_data = []
    
    for dim in dimensions:
        for p in PERCENTAGES:
            res = analyze_high_spread(dim, config, top_percent=p)
            if res is None:
                continue
            
            if p == PERCENTAGES[0]:
                summary_data.append([dim.upper(), "Overall", 
                                   res['overall']['acc_std'], res['overall']['acc_qr'], 
                                   res['overall']['auc_std'], res['overall']['auc_qr']])
            summary_data.append([dim.upper(), f"Top {p}% Spread", 
                               res['subset']['acc_std'], res['subset']['acc_qr'], 
                               res['subset']['auc_std'], res['subset']['auc_qr']])
    
    # Summary Table
    print("\n" + "="*100)
    print(f"{'RM Performance Summary Table':^100}")
    print("="*100)
    print(f"{'Dim':<12} | {'Range':<20} | {'Std Acc':>8} | {'QR Acc':>8} | {'Std AUC':>8} | {'QR AUC':>8} | {'QR Gap'}")
    print("-" * 100)
    for row in summary_data:
        dim, rng, acc_s, acc_q, auc_s, auc_q = row
        gap = (acc_q - acc_s) * 100
        print(f"{dim:<12} | {rng:<20} | {acc_s:8.4f} | {acc_q:8.4f} | {auc_s:8.4f} | {auc_q:8.4f} | {gap:+6.2f}%p")
    print("="*100)


if __name__ == "__main__":
    main()
