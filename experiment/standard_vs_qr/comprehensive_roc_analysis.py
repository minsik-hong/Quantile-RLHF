"""
Comprehensive ROC Analysis: Standard vs QR

Usage:
    python comprehensive_roc_analysis.py --dataset pku
    python comprehensive_roc_analysis.py --dataset helpsteer
    python comprehensive_roc_analysis.py --dataset hummer
"""

import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import entropy

from config import get_dataset_config, get_conflict_dims, get_output_dir, add_dataset_argument


def compute_metrics(q1, q2, n_bins=50):
    """두 분포 사이의 KL 및 JS Divergence 계산"""
    def local_norm(x): return (x - x.mean()) / (x.std() + 1e-8)
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
    
    return kl_div, js_div


def load_and_prepare_data(config):
    """데이터 로드 및 준비"""
    dimensions = config["dimensions"]
    standard_paths = config["standard_paths"]
    qr_paths = config["qr_paths"]
    
    data = {}
    for dim in dimensions:
        std_path = standard_paths.get(dim)
        qr_path = qr_paths.get(dim)
        
        if not std_path or not qr_path:
            print(f"[SKIP] {dim}: paths not configured")
            continue
            
        if not os.path.exists(std_path) or not os.path.exists(qr_path):
            print(f"[SKIP] {dim}: paths not found")
            continue
        
        print(f"Loading {dim} data...")
        try:
            std_h = pd.read_csv(os.path.join(std_path, "higher_rewards.csv"))["reward"]
            std_l = pd.read_csv(os.path.join(std_path, "lower_rewards.csv"))["reward"]
            qr_h = pd.read_csv(os.path.join(qr_path, "higher_quantiles.csv"))
            qr_l = pd.read_csv(os.path.join(qr_path, "lower_quantiles.csv"))
        except Exception as e:
            print(f"[ERROR] {dim}: {e}")
            continue
        
        qr_h_mean = qr_h.mean(axis=1)
        qr_l_mean = qr_l.mean(axis=1)
        
        # Spread (Hard Case)
        spread_h = qr_h['q9'] - qr_h['q0'] if 'q9' in qr_h.columns and 'q0' in qr_h.columns else pd.Series([0]*len(qr_h))
        spread_l = qr_l['q9'] - qr_l['q0'] if 'q9' in qr_l.columns and 'q0' in qr_l.columns else pd.Series([0]*len(qr_l))
        spread = (spread_h + spread_l) / 2
        
        data[dim] = {
            'std_h': std_h, 'std_l': std_l,
            'qr_h': qr_h, 'qr_l': qr_l,
            'qr_h_mean': qr_h_mean, 'qr_l_mean': qr_l_mean,
            'spread': spread
        }
    
    return data


def add_conflict_metrics(data, conflict_dims):
    """Conflict 메트릭 추가 (KL/JS Divergence)"""
    if not conflict_dims or len(conflict_dims) != 2:
        return data
    
    dim1, dim2 = conflict_dims
    if dim1 not in data or dim2 not in data:
        return data
    
    print(f"Calculating KL & JS Divergence for conflict analysis between {dim1} and {dim2}...")
    
    kl_scores, js_scores = [], []
    min_len = min(len(data[dim1]['qr_h']), len(data[dim2]['qr_h']))
    
    for i in range(min_len):
        kl_h, js_h = compute_metrics(data[dim1]['qr_h'].iloc[i].values, data[dim2]['qr_h'].iloc[i].values)
        kl_l, js_l = compute_metrics(data[dim1]['qr_l'].iloc[i].values, data[dim2]['qr_l'].iloc[i].values)
        kl_scores.append((kl_h + kl_l) / 2.0)
        js_scores.append((js_h + js_l) / 2.0)
    
    for dim in data:
        if len(data[dim]['std_h']) == min_len:
            data[dim]['kl_conflict'] = pd.Series(kl_scores)
            data[dim]['js_conflict'] = pd.Series(js_scores)
    
    return data


def plot_roc_subplot(ax, h_std, l_std, h_qr, l_qr, title):
    """ROC 서브플롯"""
    y_true = np.concatenate([np.ones(len(h_std)), np.zeros(len(l_std))])
    
    y_score_std = np.concatenate([h_std, l_std])
    fpr_s, tpr_s, _ = roc_curve(y_true, y_score_std)
    auc_s = auc(fpr_s, tpr_s)
    
    y_score_qr = np.concatenate([h_qr, l_qr])
    fpr_q, tpr_q, _ = roc_curve(y_true, y_score_qr)
    auc_q = auc(fpr_q, tpr_q)
    
    ax.plot(fpr_s, tpr_s, label=f'Std (AUC={auc_s:.4f})', color='blue', alpha=0.6, lw=1.5)
    ax.plot(fpr_q, tpr_q, label=f'QR (AUC={auc_q:.4f})', color='red', alpha=0.8, lw=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('FPR', fontsize=8)
    ax.set_ylabel('TPR', fontsize=8)
    ax.legend(fontsize='x-small', loc='lower right')
    ax.grid(alpha=0.2)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive ROC Analysis")
    add_dataset_argument(parser)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Comprehensive ROC Analysis - Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    config = get_dataset_config(args.dataset, auto_latest=getattr(args, "auto_latest", False))
    data = load_and_prepare_data(config)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    # Conflict 메트릭 추가
    conflict_dims = get_conflict_dims(args.dataset)
    data = add_conflict_metrics(data, conflict_dims)
    
    dimensions = list(data.keys())
    n_dims = len(dimensions)
    
    if n_dims == 0:
        print("No dimensions to analyze. Exiting.")
        return
    
    fig, axes = plt.subplots(n_dims, 10, figsize=(38, 5 * n_dims), squeeze=False)
    percentages = [10, 5, 1]
    
    for i, dim in enumerate(dimensions):
        d = data[dim]
        
        # 1. Overall
        plot_roc_subplot(axes[i, 0], d['std_h'], d['std_l'], d['qr_h_mean'], d['qr_l_mean'], 
                        f"{dim.upper()}\nOverall")
        
        # 2. Spread (Col 1-3)
        for j, p in enumerate(percentages):
            mask = d['spread'] >= d['spread'].quantile(1 - p/100.0)
            if mask.sum() > 0:
                plot_roc_subplot(axes[i, 1+j], d['std_h'][mask], d['std_l'][mask], 
                                d['qr_h_mean'][mask], d['qr_l_mean'][mask], 
                                f"{dim.upper()}\nSpread Top {p}%")
            else:
                axes[i, 1+j].axis('off')
        
        # 3. KL Conflict (Col 4-6)
        if 'kl_conflict' in d:
            for j, p in enumerate(percentages):
                mask = d['kl_conflict'] >= d['kl_conflict'].quantile(1 - p/100.0)
                if mask.sum() > 0:
                    plot_roc_subplot(axes[i, 4+j], d['std_h'][mask], d['std_l'][mask], 
                                    d['qr_h_mean'][mask], d['qr_l_mean'][mask], 
                                    f"{dim.upper()}\nKL Top {p}%")
                else:
                    axes[i, 4+j].axis('off')
        else:
            for j in range(3):
                axes[i, 4+j].axis('off')
        
        # 4. JS Conflict (Col 7-9)
        if 'js_conflict' in d:
            for j, p in enumerate(percentages):
                mask = d['js_conflict'] >= d['js_conflict'].quantile(1 - p/100.0)
                if mask.sum() > 0:
                    plot_roc_subplot(axes[i, 7+j], d['std_h'][mask], d['std_l'][mask], 
                                    d['qr_h_mean'][mask], d['qr_l_mean'][mask], 
                                    f"{dim.upper()}\nJS Top {p}%")
                else:
                    axes[i, 7+j].axis('off')
        else:
            for j in range(3):
                axes[i, 7+j].axis('off')
    
    dataset_name = config['name']
    plt.suptitle(f"Comprehensive ROC Analysis: Standard vs QR ({dataset_name})", fontsize=22, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_dir = get_output_dir(args.dataset)
    save_path = os.path.join(save_dir, f"comprehensive_roc_analysis_{args.dataset}.png")
    plt.savefig(save_path, dpi=120)
    print(f"\nAnalysis complete. Plot saved to {save_path}")


if __name__ == "__main__":
    main()
