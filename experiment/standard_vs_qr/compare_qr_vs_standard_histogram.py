"""
Standard vs QR Histogram Comparison

Usage:
    python compare_qr_vs_standard_histogram.py --dataset pku
    python compare_qr_vs_standard_histogram.py --dataset helpsteer
    python compare_qr_vs_standard_histogram.py --dataset hummer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance
import os
import argparse

from config import get_dataset_config, get_output_dir, add_dataset_argument


def compute_distribution_distances(higher, lower, n_bins=100):
    """두 분포 간의 거리 지표들 계산"""
    min_val = min(higher.min(), lower.min())
    max_val = max(higher.max(), lower.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    hist_h, _ = np.histogram(higher, bins=bins, density=True)
    hist_l, _ = np.histogram(lower, bins=bins, density=True)
    
    eps = 1e-10
    hist_h = hist_h + eps
    hist_l = hist_l + eps
    hist_h = hist_h / hist_h.sum()
    hist_l = hist_l / hist_l.sum()
    
    kl_h_l = entropy(hist_h, hist_l)
    kl_l_h = entropy(hist_l, hist_h)
    m = 0.5 * (hist_h + hist_l)
    js_div = 0.5 * entropy(hist_h, m) + 0.5 * entropy(hist_l, m)
    wasserstein = wasserstein_distance(higher, lower)
    bc = np.sum(np.sqrt(hist_h * hist_l))
    bhattacharyya = -np.log(bc + eps)
    
    return {
        "KL(H||L)": kl_h_l,
        "KL(L||H)": kl_l_h,
        "JS Div": js_div,
        "Wasserstein": wasserstein,
        "Bhattacharyya": bhattacharyya,
    }


def load_data(config):
    """데이터 로드"""
    dimensions = config["dimensions"]
    standard_paths = config["standard_paths"]
    qr_paths = config["qr_paths"]
    
    data = {}
    loaded_dims = []
    
    for dim in dimensions:
        std_path = standard_paths.get(dim)
        qr_path = qr_paths.get(dim)
        
        if not std_path or not qr_path:
            print(f"[SKIP] {dim}: paths not configured")
            continue
            
        if not os.path.exists(std_path) or not os.path.exists(qr_path):
            print(f"[SKIP] {dim}: paths not found")
            continue
        
        try:
            std_higher = pd.read_csv(os.path.join(std_path, "higher_rewards.csv"))["reward"]
            std_lower = pd.read_csv(os.path.join(std_path, "lower_rewards.csv"))["reward"]
            
            # QR expectation 파일이 없으면 mean 계산
            qr_exp_higher_path = os.path.join(qr_path, "higher_quantiles_expectation.csv")
            qr_exp_lower_path = os.path.join(qr_path, "lower_quantiles_expectation.csv")
            
            qr_higher_full = pd.read_csv(os.path.join(qr_path, "higher_quantiles.csv"))
            qr_lower_full = pd.read_csv(os.path.join(qr_path, "lower_quantiles.csv"))
            
            if os.path.exists(qr_exp_higher_path):
                qr_higher = pd.read_csv(qr_exp_higher_path)["q_expectation"]
            else:
                qr_higher = qr_higher_full.mean(axis=1)
            
            if os.path.exists(qr_exp_lower_path):
                qr_lower = pd.read_csv(qr_exp_lower_path)["q_expectation"]
            else:
                qr_lower = qr_lower_full.mean(axis=1)
            
            data[dim] = {
                "std_higher": std_higher,
                "std_lower": std_lower,
                "qr_higher": qr_higher,
                "qr_lower": qr_lower,
                "qr_higher_q0": qr_higher_full["q0"] if "q0" in qr_higher_full.columns else qr_higher_full.iloc[:, 0],
                "qr_lower_q0": qr_lower_full["q0"] if "q0" in qr_lower_full.columns else qr_lower_full.iloc[:, 0],
                "qr_higher_q9": qr_higher_full["q9"] if "q9" in qr_higher_full.columns else qr_higher_full.iloc[:, -1],
                "qr_lower_q9": qr_lower_full["q9"] if "q9" in qr_lower_full.columns else qr_lower_full.iloc[:, -1],
            }
            loaded_dims.append(dim)
            print(f"[LOADED] {dim}")
        except Exception as e:
            print(f"[ERROR] {dim}: {e}")
    
    return data, loaded_dims


def plot_histogram_grid(data_dict, dimensions, title, out_path, density=False, shared_axis=False, labels=("Higher", "Lower")):
    """공통 히스토그램 그리기 함수"""
    n_rows = max(k[0] for k in data_dict.keys()) + 1
    n_cols = max(k[1] for k in data_dict.keys()) + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows), squeeze=False)
    
    colors = {"higher": "#2ecc71", "lower": "#e74c3c"}
    
    if shared_axis:
        all_vals = []
        for k, v in data_dict.items():
            all_vals.extend([v["higher"], v["lower"]])
        all_concat = pd.concat(all_vals)
        x_min, x_max = all_concat.min(), all_concat.max()
        bins = np.linspace(x_min, x_max, 61)
        
        y_max = 0
        for k, v in data_dict.items():
            for arr in [v["higher"], v["lower"]]:
                counts, _ = np.histogram(arr, bins=bins, density=density)
                y_max = max(y_max, counts.max())
        y_max = y_max * 1.1
    else:
        bins = 60
        x_min, x_max, y_max = None, None, None
    
    for (row, col), v in data_dict.items():
        ax = axes[row, col]
        higher = v["higher"]
        lower = v["lower"]
        cell_title = v["title"]
        
        ax.hist(higher, bins=bins, alpha=0.7, density=density,
                label=f"{labels[0]} (μ={higher.mean():.2f})", color=colors["higher"], 
                edgecolor="black", linewidth=0.5)
        ax.hist(lower, bins=bins, alpha=0.7, density=density,
                label=f"{labels[1]} (μ={lower.mean():.2f})", color=colors["lower"], 
                edgecolor="black", linewidth=0.5)
        ax.axvline(higher.mean(), color="darkgreen", linestyle="--", linewidth=2)
        ax.axvline(lower.mean(), color="darkred", linestyle="--", linewidth=2)
        
        if shared_axis:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max)
        
        ax.set_xlabel("Reward", fontsize=12)
        ax.set_ylabel("Probability Density" if density else "Count", fontsize=12)
        ax.set_title(cell_title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out_path}")
    plt.close()


def make_std_qr_data_dict(data, dimensions):
    """Standard vs QR 데이터 딕셔너리 생성"""
    res = {}
    for i, dim in enumerate(dimensions):
        if dim not in data:
            continue
        res[(0, i)] = {"higher": data[dim]["std_higher"], "lower": data[dim]["std_lower"], 
                       "title": f"Standard RM - {dim.capitalize()}"}
        res[(1, i)] = {"higher": data[dim]["qr_higher"], "lower": data[dim]["qr_lower"], 
                       "title": f"QR RM - {dim.capitalize()}"}
    return res


def plot_q0_vs_q9_by_dimension(data, dimensions, out_path, density=False, shared_axis=False):
    """DIMENSIONS 각각에 대해 Higher/Lower의 q0 vs q9 비교"""
    n_dims = len(dimensions)
    fig, axes = plt.subplots(1, n_dims, figsize=(7*n_dims, 5), squeeze=False)
    
    colors = {"higher_q9": "#2ecc71", "higher_q0": "#27ae60", 
              "lower_q9": "#e74c3c", "lower_q0": "#c0392b"}
    
    if shared_axis:
        all_vals = []
        for dim in dimensions:
            if dim not in data:
                continue
            all_vals.extend([data[dim]["qr_higher_q0"], data[dim]["qr_higher_q9"],
                           data[dim]["qr_lower_q0"], data[dim]["qr_lower_q9"]])
        if all_vals:
            all_concat = pd.concat(all_vals)
            x_min, x_max = all_concat.min(), all_concat.max()
            bins = np.linspace(x_min, x_max, 61)
            
            y_max = 0
            for dim in dimensions:
                if dim not in data:
                    continue
                for arr in [data[dim]["qr_higher_q0"], data[dim]["qr_higher_q9"],
                           data[dim]["qr_lower_q0"], data[dim]["qr_lower_q9"]]:
                    counts, _ = np.histogram(arr, bins=bins, density=density)
                    y_max = max(y_max, counts.max())
            y_max = y_max * 1.1
        else:
            bins = 60
            x_min, x_max, y_max = None, None, None
    else:
        bins = 60
        x_min, x_max, y_max = None, None, None
    
    for idx, dim in enumerate(dimensions):
        ax = axes[0, idx]
        
        if dim not in data:
            ax.axis('off')
            continue
        
        higher_q0 = data[dim]["qr_higher_q0"]
        higher_q9 = data[dim]["qr_higher_q9"]
        lower_q0 = data[dim]["qr_lower_q0"]
        lower_q9 = data[dim]["qr_lower_q9"]
        
        ax.hist(higher_q9, bins=bins, alpha=0.6, density=density,
                label=f"Higher q9 (μ={higher_q9.mean():.2f})", color=colors["higher_q9"], edgecolor="black", linewidth=0.3)
        ax.hist(higher_q0, bins=bins, alpha=0.6, density=density,
                label=f"Higher q0 (μ={higher_q0.mean():.2f})", color=colors["higher_q0"], edgecolor="black", linewidth=0.3)
        ax.hist(lower_q9, bins=bins, alpha=0.6, density=density,
                label=f"Lower q9 (μ={lower_q9.mean():.2f})", color=colors["lower_q9"], edgecolor="black", linewidth=0.3)
        ax.hist(lower_q0, bins=bins, alpha=0.6, density=density,
                label=f"Lower q0 (μ={lower_q0.mean():.2f})", color=colors["lower_q0"], edgecolor="black", linewidth=0.3)
        
        ax.axvline(higher_q9.mean(), color="#1e8449", linestyle="--", linewidth=2)
        ax.axvline(higher_q0.mean(), color="#196f3d", linestyle=":", linewidth=2)
        ax.axvline(lower_q9.mean(), color="#922b21", linestyle="--", linewidth=2)
        ax.axvline(lower_q0.mean(), color="#7b241c", linestyle=":", linewidth=2)
        
        if shared_axis and x_min is not None:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max)
        
        ax.set_xlabel("Reward", fontsize=12)
        ax.set_ylabel("Probability Density" if density else "Count", fontsize=12)
        ax.set_title(f"{dim.capitalize()} - q0 vs q9", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out_path}")
    plt.close()


def compute_and_print_distances(data, dimensions):
    """KL Divergence 및 기타 분포 거리 지표 계산"""
    print("\n" + "="*80)
    print("Distribution Distance Metrics: Higher vs Lower")
    print("="*80)
    
    results = {}
    for model_type in ["Standard", "QR"]:
        for dim in dimensions:
            if dim not in data:
                continue
            d = data[dim]
            if model_type == "Standard":
                higher = d["std_higher"].values
                lower = d["std_lower"].values
            elif model_type == "QR":
                higher = d["qr_higher"].values
                lower = d["qr_lower"].values
            
            distances = compute_distribution_distances(higher, lower)
            key = f"{model_type}_{dim}"
            results[key] = distances
            
            print(f"\n📊 {model_type} - {dim.capitalize()}")
            print("-"*50)
            for metric, val in distances.items():
                print(f"  {metric:15s}: {val:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Standard vs QR Histogram Comparison")
    add_dataset_argument(parser)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Histogram Comparison - Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    config = get_dataset_config(args.dataset, auto_latest=getattr(args, "auto_latest", False))
    data, loaded_dims = load_data(config)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    save_dir = get_output_dir(args.dataset)
    
    # Standard vs QR
    std_qr_dict = make_std_qr_data_dict(data, loaded_dims)
    
    if std_qr_dict:
        plot_histogram_grid(std_qr_dict, loaded_dims, f"Higher vs Lower: Standard vs QR ({config['name']})", 
                           os.path.join(save_dir, f"histogram_comparison_{args.dataset}.png"), 
                           density=False, shared_axis=False)
        plot_histogram_grid(std_qr_dict, loaded_dims, f"Probability Density: Standard vs QR ({config['name']})", 
                           os.path.join(save_dir, f"histogram_density_{args.dataset}.png"), 
                           density=True, shared_axis=False)
    
    # q0 vs q9
    plot_q0_vs_q9_by_dimension(data, loaded_dims, 
                               os.path.join(save_dir, f"q0_q9_histogram_{args.dataset}.png"), 
                               density=False, shared_axis=False)
    plot_q0_vs_q9_by_dimension(data, loaded_dims, 
                               os.path.join(save_dir, f"q0_q9_histogram_density_{args.dataset}.png"), 
                               density=True, shared_axis=False)
    
    # 통계
    compute_and_print_distances(data, loaded_dims)
    
    print(f"\n✅ Analysis complete. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
