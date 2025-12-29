"""
Standard vs QR KDE (Kernel Density Estimation) Comparison

Usage:
    python compare_qr_vs_standard_kde.py --dataset pku
    python compare_qr_vs_standard_kde.py --dataset helpsteer
    python compare_qr_vs_standard_kde.py --dataset hummer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy, wasserstein_distance
import os
import argparse

from config import get_dataset_config, get_output_dir, add_dataset_argument


def compute_kde_distances(higher, lower, n_points=1000):
    """KDE를 사용하여 두 분포 간의 거리 지표 계산"""
    kde_h = gaussian_kde(higher)
    kde_l = gaussian_kde(lower)
    
    min_val = min(higher.min(), lower.min())
    max_val = max(higher.max(), lower.max())
    x_grid = np.linspace(min_val, max_val, n_points)
    
    pdf_h = kde_h(x_grid)
    pdf_l = kde_l(x_grid)
    
    p = pdf_h / pdf_h.sum()
    q = pdf_l / pdf_l.sum()
    
    kl_h_l = entropy(p, q)
    kl_l_h = entropy(q, p)
    m = 0.5 * (p + q)
    js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    wasserstein = wasserstein_distance(higher, lower)
    bc = np.sum(np.sqrt(p * q))
    bhattacharyya = -np.log(bc + 1e-10)
    
    return {
        "KL(H||L)": kl_h_l,
        "KL(L||H)": kl_l_h,
        "JS Div": js_div,
        "Wasserstein": wasserstein,
        "Bhattacharyya": bhattacharyya,
    }, (x_grid, pdf_h, pdf_l)


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
            continue
        if not os.path.exists(std_path) or not os.path.exists(qr_path):
            print(f"[SKIP] {dim}: paths not found")
            continue
        
        try:
            std_higher = pd.read_csv(os.path.join(std_path, "higher_rewards.csv"))["reward"]
            std_lower = pd.read_csv(os.path.join(std_path, "lower_rewards.csv"))["reward"]
            
            # QR expectation
            qr_exp_higher_path = os.path.join(qr_path, "higher_quantiles_expectation.csv")
            qr_exp_lower_path = os.path.join(qr_path, "lower_quantiles_expectation.csv")
            
            if os.path.exists(qr_exp_higher_path):
                qr_higher = pd.read_csv(qr_exp_higher_path)["q_expectation"]
            else:
                qr_higher = pd.read_csv(os.path.join(qr_path, "higher_quantiles.csv")).mean(axis=1)
            
            if os.path.exists(qr_exp_lower_path):
                qr_lower = pd.read_csv(qr_exp_lower_path)["q_expectation"]
            else:
                qr_lower = pd.read_csv(os.path.join(qr_path, "lower_quantiles.csv")).mean(axis=1)
            
            data[dim] = {
                "std_higher": std_higher,
                "std_lower": std_lower,
                "qr_higher": qr_higher,
                "qr_lower": qr_lower,
            }
            loaded_dims.append(dim)
            print(f"[LOADED] {dim}")
        except Exception as e:
            print(f"[ERROR] {dim}: {e}")
    
    return data, loaded_dims


def plot_kde_comparison(data, dimensions, results_info, title, out_path):
    """KDE 기반 분포 시각화"""
    n_dims = len(dimensions)
    fig, axes = plt.subplots(2, n_dims, figsize=(7*n_dims, 10), squeeze=False)
    
    model_types = ["Standard", "QR"]
    colors = {"higher": "#3498db", "lower": "#e74c3c"}
    
    # 전역축 범위 계산
    all_x_grids = []
    all_pdfs = []
    for key in results_info:
        x_grid, pdf_h, pdf_l = results_info[key]["grid_data"]
        all_x_grids.append(x_grid)
        all_pdfs.extend([pdf_h, pdf_l])
    
    x_min = min([grid.min() for grid in all_x_grids])
    x_max = max([grid.max() for grid in all_x_grids])
    y_max = max([pdf.max() for pdf in all_pdfs]) * 1.1
    
    for r, m_type in enumerate(model_types):
        m_type_key = "std" if m_type == "Standard" else "qr"
        for c, dim in enumerate(dimensions):
            ax = axes[r, c]
            
            key = f"{m_type}_{dim}"
            if key not in results_info:
                ax.axis('off')
                continue
            
            x_grid, pdf_h, pdf_l = results_info[key]["grid_data"]
            
            ax.plot(x_grid, pdf_h, label=f"Higher (μ={data[dim][f'{m_type_key}_higher'].mean():.2f})", 
                    color=colors["higher"], linewidth=2.5)
            ax.plot(x_grid, pdf_l, label=f"Lower (μ={data[dim][f'{m_type_key}_lower'].mean():.2f})", 
                    color=colors["lower"], linewidth=2.5)
            
            ax.fill_between(x_grid, pdf_h, alpha=0.2, color=colors["higher"])
            ax.fill_between(x_grid, pdf_l, alpha=0.2, color=colors["lower"])
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max)
            
            ax.set_title(f"{m_type} RM - {dim.capitalize()}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Reward", fontsize=12)
            ax.set_ylabel("Density (KDE)", fontsize=12)
            ax.legend(fontsize=10, loc='upper left')
            ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Standard vs QR KDE Comparison")
    add_dataset_argument(parser)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"KDE Comparison - Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    config = get_dataset_config(args.dataset, auto_latest=getattr(args, "auto_latest", False))
    data, loaded_dims = load_data(config)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    all_results = {}
    
    print("\n" + "="*80)
    print("KDE-based Distribution Distance Metrics")
    print("="*80)
    
    for m_type in ["Standard", "QR"]:
        m_type_key = "std" if m_type == "Standard" else "qr"
        for dim in loaded_dims:
            higher = data[dim][f"{m_type_key}_higher"]
            lower = data[dim][f"{m_type_key}_lower"]
            
            distances, grid_info = compute_kde_distances(higher, lower)
            key = f"{m_type}_{dim}"
            all_results[key] = {"metrics": distances, "grid_data": grid_info}
            
            print(f"\n📊 {key}")
            for m, v in distances.items():
                print(f"  {m:15s}: {v:.4f}")
    
    # 시각화
    save_dir = get_output_dir(args.dataset)
    plot_kde_comparison(data, loaded_dims, all_results, 
                       f"Reward Distribution Comparison (KDE) - {config['name']}", 
                       os.path.join(save_dir, f"reward_kde_comparison_{args.dataset}.png"))
    
    print(f"\n✅ Analysis complete. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
