"""
Distributional Conflict Analysis: JS & KL Divergence (Helpful vs Safe)

분포의 형태적 차이를 측정하는 JS Divergence와 정보 이론적 차이를 측정하는 KL Divergence를 사용하여 분석.
1. 각 차원을 Z-score 정규화하여 형태적 특징 추출.
2. 동일 답변에 대한 JS 및 KL Divergence 계산 (H-H, L-L).
3. 각 지표별 상위 충돌 구간에서 QR과 Standard의 성능 비교 (Acc & AUC).

Note: 이 분석은 PKU-SafeRLHF 데이터셋에서만 의미가 있습니다 (helpful vs safe).

Usage:
    python conflict_analysis.py --dataset pku
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score

from config import get_dataset_config, get_conflict_dims, add_dataset_argument


def compute_divergences(q1, q2, n_bins=50):
    """두 분위수 집합 사이의 JS 및 KL Divergence 계산"""
    def local_norm(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    
    q1_n, q2_n = local_norm(q1), local_norm(q2)
    
    min_v = min(q1_n.min(), q2_n.min())
    max_v = max(q1_n.max(), q2_n.max())
    bins = np.linspace(min_v, max_v, n_bins + 1)
    
    p, _ = np.histogram(q1_n, bins=bins, density=True)
    q, _ = np.histogram(q2_n, bins=bins, density=True)
    
    p = (p + 1e-10) / (p.sum() + 1e-10 * n_bins)
    q = (q + 1e-10) / (q.sum() + 1e-10 * n_bins)
    
    kl_div = entropy(p, q)
    m = 0.5 * (p + q)
    js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    
    return js_div, kl_div


def compute_auc(h_scores, l_scores):
    """AUC 계산"""
    y_true = np.concatenate([np.ones(len(h_scores)), np.zeros(len(l_scores))])
    y_score = np.concatenate([h_scores, l_scores])
    return roc_auc_score(y_true, y_score)


def analyze_conflicts(config, conflict_dims):
    """충돌 분석"""
    if not conflict_dims or len(conflict_dims) != 2:
        print("Conflict analysis requires exactly 2 dimensions (e.g., helpful vs safe)")
        return
    
    dim1, dim2 = conflict_dims
    dimensions = [dim1, dim2]
    
    standard_paths = config["standard_paths"]
    qr_paths = config["qr_paths"]
    
    # 데이터 로드
    data_all = {}
    for dim in dimensions:
        std_path = standard_paths.get(dim)
        qr_path = qr_paths.get(dim)
        
        if not std_path or not qr_path:
            print(f"[ERROR] {dim}: paths not configured")
            return
        if not os.path.exists(std_path) or not os.path.exists(qr_path):
            print(f"[ERROR] {dim}: paths not found")
            return
        
        try:
            qr_h = pd.read_csv(os.path.join(qr_path, "higher_quantiles.csv"))
            qr_l = pd.read_csv(os.path.join(qr_path, "lower_quantiles.csv"))
            std_h = pd.read_csv(os.path.join(std_path, "higher_rewards.csv"))["reward"]
            std_l = pd.read_csv(os.path.join(std_path, "lower_rewards.csv"))["reward"]
            
            data_all[dim] = {
                'qr_h': qr_h, 'qr_l': qr_l,
                'std_h': std_h, 'std_l': std_l,
                'qr_h_mean': qr_h.mean(axis=1), 'qr_l_mean': qr_l.mean(axis=1)
            }
            print(f"[LOADED] {dim}")
        except Exception as e:
            print(f"[ERROR] {dim}: {e}")
            return
    
    # Divergence 계산
    print(f"\nCalculating JS and KL Divergences between {dim1} and {dim2}...")
    js_scores, kl_scores = [], []
    
    min_len = min(len(data_all[dim1]['qr_h']), len(data_all[dim2]['qr_h']))
    
    for i in range(min_len):
        js_h, kl_h = compute_divergences(
            data_all[dim1]['qr_h'].iloc[i].values, 
            data_all[dim2]['qr_h'].iloc[i].values
        )
        js_l, kl_l = compute_divergences(
            data_all[dim1]['qr_l'].iloc[i].values, 
            data_all[dim2]['qr_l'].iloc[i].values
        )
        js_scores.append((js_h + js_l) / 2.0)
        kl_scores.append((kl_h + kl_l) / 2.0)
    
    # 분석 및 결과 출력
    for metric in ['kl', 'js']:
        summary = []
        col_scores = kl_scores if metric == 'kl' else js_scores
        df_scores = pd.DataFrame({'conflict': col_scores})
        
        def add_to_summary(dim_name, range_name, mask=None):
            d = data_all[dim_name.lower()]
            if mask is not None:
                h = d['std_h'][mask]
                l = d['std_l'][mask]
                qh = d['qr_h_mean'][mask]
                ql = d['qr_l_mean'][mask]
            else:
                h, l, qh, ql = d['std_h'][:min_len], d['std_l'][:min_len], \
                               d['qr_h_mean'][:min_len], d['qr_l_mean'][:min_len]
            
            if len(h) == 0:
                return
            
            s_acc = (h.values > l.values).mean()
            q_acc = (qh.values > ql.values).mean()
            s_auc = compute_auc(h, l)
            q_auc = compute_auc(qh, ql)
            summary.append([dim_name, range_name, s_acc, q_acc, s_auc, q_auc])
        
        for dim in dimensions:
            add_to_summary(dim.upper(), "Overall")
            for p in [10, 5, 1]:
                thresh = df_scores['conflict'].quantile(1.0 - p/100.0)
                mask = df_scores['conflict'] >= thresh
                add_to_summary(dim.upper(), f"Top {p}% {metric.upper()}", mask)
        
        print("\n" + "="*100)
        print(f"{f'RM Performance Summary Table ({metric.upper()} Conflict: {dim1} vs {dim2})':^100}")
        print("="*100)
        print(f"{'Dim':<10} | {'Range':<20} | {'Std Acc':>8} | {'QR Acc':>8} | {'Std AUC':>8} | {'QR AUC':>8} | {'QR Gap'}")
        print("-" * 100)
        for row in summary:
            dim_name, rng, s_acc, q_acc, s_auc, q_auc = row
            gap = (q_acc - s_acc) * 100
            print(f"{dim_name:<10} | {rng:<20} | {s_acc:8.4f} | {q_acc:8.4f} | {s_auc:8.4f} | {q_auc:8.4f} | {gap:+6.2f}%p")
        print("="*100)


def main():
    parser = argparse.ArgumentParser(description="Conflict Analysis (Helpful vs Safe)")
    add_dataset_argument(parser)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Conflict Analysis - Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    config = get_dataset_config(args.dataset, auto_latest=getattr(args, "auto_latest", False))
    conflict_dims = get_conflict_dims(args.dataset)
    
    if not conflict_dims:
        print(f"⚠️  Conflict analysis is only available for PKU dataset (helpful vs safe).")
        print(f"   Dataset '{args.dataset}' does not have conflict dimensions configured.")
        return
    
    print(f"Conflict dimensions: {conflict_dims[0]} vs {conflict_dims[1]}")
    
    analyze_conflicts(config, conflict_dims)
    
    print(f"\n✅ Conflict analysis complete.")


if __name__ == "__main__":
    main()
