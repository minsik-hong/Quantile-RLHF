"""
Comprehensive Statistical Analysis: QR vs Standard Reward Model

Usage:
    python comprehensive_statistical_analysis.py --dataset pku
    python comprehensive_statistical_analysis.py --dataset helpsteer
    python comprehensive_statistical_analysis.py --dataset hummer
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
import argparse

from config import get_dataset_config, get_output_dir, add_dataset_argument


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def compute_dprime(h, l):
    return (np.mean(h) - np.mean(l)) / np.sqrt(0.5 * (np.var(h) + np.var(l)))


def optimal_accuracy(h, l):
    labels = np.concatenate([np.ones(len(h)), np.zeros(len(l))])
    scores = np.concatenate([h, l])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    return np.mean((scores >= best_thresh) == labels)


def compute_distances(h, l, n_bins=100):
    min_v, max_v = min(h.min(), l.min()), max(h.max(), l.max())
    bins = np.linspace(min_v, max_v, n_bins + 1)
    ph, _ = np.histogram(h, bins=bins, density=True)
    pl, _ = np.histogram(l, bins=bins, density=True)
    
    eps = 1e-10
    ph = (ph + eps) / (ph + eps).sum()
    pl = (pl + eps) / (pl + eps).sum()
    
    kl_hl = entropy(ph, pl)
    kl_lh = entropy(pl, ph)
    m = 0.5 * (ph + pl)
    js_div = 0.5 * (entropy(ph, m) + entropy(pl, m))
    wass = wasserstein_distance(h, l)
    bhatt = -np.log(np.sum(np.sqrt(ph * pl)))
    
    return {
        "KL(H||L)": kl_hl, "KL(L||H)": kl_lh, "JS Div": js_div,
        "Wasserstein": wass, "Bhattacharyya": bhatt
    }


def plot_performance_comparison(results, dimensions, save_dir, dataset_name):
    """성능 비교 시각화"""
    metrics = ["Cohen's d", "d-prime", "AUC-ROC", "Accuracy"]
    
    keys = []
    labels = []
    for m_type in ["Standard", "QR"]:
        for dim in dimensions:
            keys.append(f"{m_type}_{dim}")
            labels.append(f"{'Std' if m_type == 'Standard' else 'QR'} {dim.capitalize()}")
    
    base_colors = ["#7ebdec", "#4a89b8", "#f39c4b", "#c0504d", "#76b7b2", "#edc948", 
                   "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#59a14f", "#e15759"]
    colors = [base_colors[i % len(base_colors)] for i in range(len(keys))]
    
    x = np.arange(len(metrics))
    n_groups = len(keys)
    width = 0.8 / n_groups
    
    fig, ax = plt.subplots(figsize=(18, 9))
    
    for i, key in enumerate(keys):
        if key not in results:
            continue
        values = [results[key][m] for m in metrics]
        offset = (i - (n_groups - 1) / 2) * width
        rects = ax.bar(x + offset, values, width, label=labels[i], color=colors[i], 
                      edgecolor='white', linewidth=0.5)
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax.set_title(f'Performance Comparison: Standard vs. QR ({dataset_name})', 
                fontsize=18, fontweight='bold', pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True, ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    all_vals = []
    for k in keys:
        if k in results:
            all_vals.extend([results[k][m] for m in metrics])
    if all_vals:
        ax.set_ylim(0, max(all_vals) * 1.15)
    
    out_path = os.path.join(save_dir, f"comprehensive_metrics_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"\n[SAVED] {out_path}")
    plt.close()


def run_analysis(config, save_dir):
    """메인 분석 함수"""
    dimensions = config["dimensions"]
    standard_paths = config["standard_paths"]
    qr_paths = config["qr_paths"]
    
    results = {}
    distance_results = {}
    loaded_dims = []
    
    for dim in dimensions:
        std_path = standard_paths.get(dim)
        qr_path = qr_paths.get(dim)
        
        if not std_path or not qr_path:
            continue
        if not os.path.exists(std_path) or not os.path.exists(qr_path):
            print(f"[SKIP] {dim}: paths not found")
            continue
        
        print(f"Analyzing {dim} data...")
        try:
            std_h = pd.read_csv(os.path.join(std_path, "higher_rewards.csv"))["reward"].values
            std_l = pd.read_csv(os.path.join(std_path, "lower_rewards.csv"))["reward"].values
            
            qr_exp_higher = os.path.join(qr_path, "higher_quantiles_expectation.csv")
            qr_exp_lower = os.path.join(qr_path, "lower_quantiles_expectation.csv")
            
            if os.path.exists(qr_exp_higher):
                qr_h = pd.read_csv(qr_exp_higher)["q_expectation"].values
            else:
                qr_h = pd.read_csv(os.path.join(qr_path, "higher_quantiles.csv")).mean(axis=1).values
            
            if os.path.exists(qr_exp_lower):
                qr_l = pd.read_csv(qr_exp_lower)["q_expectation"].values
            else:
                qr_l = pd.read_csv(os.path.join(qr_path, "lower_quantiles.csv")).mean(axis=1).values
            
            loaded_dims.append(dim)
        except Exception as e:
            print(f"[ERROR] {dim}: {e}")
            continue
        
        for m_type, h, l in [("Standard", std_h, std_l), ("QR", qr_h, qr_l)]:
            key = f"{m_type}_{dim}"
            
            results[key] = {
                "Cohen's d": cohens_d(h, l),
                "d-prime": compute_dprime(h, l),
                "AUC-ROC": roc_auc_score(np.concatenate([np.ones(len(h)), np.zeros(len(l))]), 
                                         np.concatenate([h, l])),
                "Accuracy": optimal_accuracy(h, l)
            }
            
            distance_results[key] = compute_distances(h, l)
    
    if not loaded_dims:
        print("No data loaded. Exiting.")
        return
    
    def print_summary_table(title, metrics_list, results_dict):
        print("\n" + "="*100)
        print(f"Summary Comparison Table ({title})")
        print("="*100)
        header = f"{'Metric':15s}"
        for m_type in ["Std", "QR"]:
            for dim in loaded_dims:
                header += f" | {m_type} {dim.capitalize()[:8]:>8s}"
        print(header)
        print("-" * len(header))
        for m in metrics_list:
            row = f"{m:15s}"
            for m_type in ["Standard", "QR"]:
                for dim in loaded_dims:
                    val = results_dict.get(f"{m_type}_{dim}", {}).get(m, 0)
                    row += f" | {val:12.4f}"
            print(row)
    
    print_summary_table("Distribution Distances", 
                       ["KL(H||L)", "KL(L||H)", "JS Div", "Wasserstein", "Bhattacharyya"], 
                       distance_results)
    print_summary_table("Performance Metrics", 
                       ["Cohen's d", "d-prime", "AUC-ROC", "Accuracy"], 
                       results)
    print("\n" + "="*100)
    
    plot_performance_comparison(results, loaded_dims, save_dir, config['name'])


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Statistical Analysis")
    add_dataset_argument(parser)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Comprehensive Statistical Analysis - Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    config = get_dataset_config(args.dataset, auto_latest=getattr(args, "auto_latest", False))
    save_dir = get_output_dir(args.dataset)
    
    run_analysis(config, save_dir)
    
    print(f"\n✅ Analysis complete. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
