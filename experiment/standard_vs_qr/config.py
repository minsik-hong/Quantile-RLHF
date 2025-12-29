"""
Standard vs QR 분석을 위한 공통 설정
데이터셋별로 경로와 차원을 관리합니다.

사용법:
    from config import get_dataset_config, AVAILABLE_DATASETS
    
    config = get_dataset_config("pku")  # 또는 "helpsteer", "hummer", "hummer-all"
    DIMENSIONS = config["dimensions"]
    STANDARD_BASE_PATHS = config["standard_base_paths"]
    QR_BASE_PATHS = config["qr_base_paths"]

데이터셋:
    - pku: PKU-SafeRLHF 2D (helpful, safe)
    - helpsteer: HelpSteer 5D (helpfulness, coherence, complexity, correctness, verbosity)
    - hummer: Hummer 6D 개별 차원 (accuracy, conciseness, depth, empathy, specificity, tone)
    - hummer-all: Hummer 6D 통합 단일 모델

자동 경로 탐색:
    --auto-latest 옵션 사용 시 eval_tables에서 최신 로그 폴더를 자동으로 찾습니다.
"""

import os
import glob
import argparse

# ============================================================
# 데이터셋 정의
# ============================================================

AVAILABLE_DATASETS = ["pku", "helpsteer", "hummer", "hummer-all"]

# Base eval_tables path
EVAL_TABLES_BASE = "/home/hail/safe-rlhf-hk/eval_tables/kms_test"


# ============================================================
# 자동 경로 탐색 함수
# ============================================================

def find_latest_log_path(log_type: str, rm_name: str, subdir: str = None) -> str:
    """
    eval_tables에서 최신 로그 폴더 자동 탐색
    
    Args:
        log_type: "reward_logs" 또는 "quantile_logs"
        rm_name: "Safe-RLHF-RM-helpful" 등
        subdir: 하위 디렉토리 (예: "alpaca-8b-reproduced-llama-3", "default")
    
    Returns:
        최신 로그 폴더 경로 또는 None
    """
    pattern = os.path.join(EVAL_TABLES_BASE, f"{log_type}_*", rm_name)
    matches = sorted(glob.glob(pattern))
    
    if not matches:
        return None
    
    latest = matches[-1]  # 날짜순 정렬되므로 마지막이 최신
    
    if subdir:
        full_path = os.path.join(latest, subdir)
        return full_path if os.path.exists(full_path) else latest
    
    return latest


def get_dimension_rm_name(dimension: str, dataset_type: str = "default") -> str:
    """차원 이름을 RM 폴더명으로 변환"""
    if dataset_type == "hummer":
        return f"Safe-RLHF-RM-hummer-{dimension}"
    else:
        return f"Safe-RLHF-RM-{dimension}"


# ============================================================
# 데이터셋별 차원 정의
# ============================================================

DATASET_DIMENSIONS = {
    "pku": ["helpful", "safe"],
    "helpsteer": ["helpfulness", "coherence", "complexity", "correctness", "verbosity"],
    "hummer": ["accuracy", "conciseness", "depth", "empathy", "specificity", "tone"],
    "hummer-all": ["all"],
}

DATASET_NAMES = {
    "pku": "PKU-SafeRLHF",
    "helpsteer": "HelpSteer",
    "hummer": "Hummer",
    "hummer-all": "Hummer-All",
}

DATASET_PATHS = {
    "pku": {
        "helpful": "/home/hail/safe-rlhf-hk/datasets/pku_saferlhf/helpful/test.jsonl",
        "safe": "/home/hail/safe-rlhf-hk/datasets/pku_saferlhf/safe/test.jsonl",
    },
    "helpsteer": {
        dim: "/home/hail/safe-rlhf-hk/datasets/helpsteer/test.jsonl"
        for dim in ["helpfulness", "coherence", "complexity", "correctness", "verbosity"]
    },
    "hummer": {
        dim: f"/home/hail/safe-rlhf-hk/datasets/hummer/preference_datasets/{dim}_preference_test.jsonl"
        for dim in ["accuracy", "conciseness", "depth", "empathy", "specificity", "tone"]
    },
    "hummer-all": {
        "all": "/home/hail/safe-rlhf-hk/datasets/hummer/preference_datasets/all_preference_test.jsonl",
    },
}


# ============================================================
# 하드코딩된 경로 (기본값, auto_latest=False일 때 사용)
# ============================================================

HARDCODED_PATHS = {
    "pku": {
        "standard": {
            "helpful": f"{EVAL_TABLES_BASE}/reward_logs_20251207_104402/Safe-RLHF-RM-helpful",
            "safe": f"{EVAL_TABLES_BASE}/reward_logs_20251207_222522/Safe-RLHF-RM-safe",
        },
        "qr": {
            "helpful": f"{EVAL_TABLES_BASE}/quantile_logs_20251214_060251/Safe-RLHF-RM-helpful",
            "safe": f"{EVAL_TABLES_BASE}/quantile_logs_20251213_035243/Safe-RLHF-RM-safe",
        },
    },
    "helpsteer": {
        "standard": {
            "helpfulness": f"{EVAL_TABLES_BASE}/reward_logs_20251204_041918/Safe-RLHF-RM-helpfulness",
            "coherence": f"{EVAL_TABLES_BASE}/reward_logs_20251203_194440/Safe-RLHF-RM-coherence",
            "complexity": f"{EVAL_TABLES_BASE}/reward_logs_20251203_210232/Safe-RLHF-RM-complexity",
            "correctness": f"{EVAL_TABLES_BASE}/reward_logs_20251203_230812/Safe-RLHF-RM-correctness",
            "verbosity": f"{EVAL_TABLES_BASE}/reward_logs_20251204_030857/Safe-RLHF-RM-verbosity",
        },
        "qr": {
            "helpfulness": f"{EVAL_TABLES_BASE}/quantile_logs_20251203_101637/Safe-RLHF-RM-helpfulness",
            "coherence": f"{EVAL_TABLES_BASE}/quantile_logs_20251203_093505/Safe-RLHF-RM-coherence",
            "complexity": f"{EVAL_TABLES_BASE}/quantile_logs_20251203_094712/Safe-RLHF-RM-complexity",
            "correctness": f"{EVAL_TABLES_BASE}/quantile_logs_20251203_104035/Safe-RLHF-RM-correctness",
            "verbosity": f"{EVAL_TABLES_BASE}/quantile_logs_20251203_111835/Safe-RLHF-RM-verbosity",
        },
    },
    "hummer": {
        "standard": {
            "accuracy": f"{EVAL_TABLES_BASE}/reward_logs_20251227_105430/Safe-RLHF-RM-hummer-accuracy",
            "conciseness": f"{EVAL_TABLES_BASE}/reward_logs_20251227_115735/Safe-RLHF-RM-hummer-conciseness",
            "depth": f"{EVAL_TABLES_BASE}/reward_logs_20251227_134928/Safe-RLHF-RM-hummer-depth",
            "empathy": f"{EVAL_TABLES_BASE}/reward_logs_20251229_114111/Safe-RLHF-RM-hummer-empathy",
            "specificity": f"{EVAL_TABLES_BASE}/reward_logs_20251227_163151/Safe-RLHF-RM-hummer-specificity",
            "tone": f"{EVAL_TABLES_BASE}/reward_logs_20251229_105314/Safe-RLHF-RM-hummer-tone",
        },
        "qr": {
            "accuracy": f"{EVAL_TABLES_BASE}/quantile_logs_20251227_013740/Safe-RLHF-RM-hummer-accuracy",
            "conciseness": f"{EVAL_TABLES_BASE}/quantile_logs_20251227_024051/Safe-RLHF-RM-hummer-conciseness",
            "depth": f"{EVAL_TABLES_BASE}/quantile_logs_20251227_043250/Safe-RLHF-RM-hummer-depth",
            "empathy": f"{EVAL_TABLES_BASE}/quantile_logs_20251227_052411/Safe-RLHF-RM-hummer-empathy",
            "specificity": f"{EVAL_TABLES_BASE}/quantile_logs_20251227_071517/Safe-RLHF-RM-hummer-specificity",
            "tone": f"{EVAL_TABLES_BASE}/quantile_logs_20251227_075350/Safe-RLHF-RM-hummer-tone",
        },
    },
    "hummer-all": {
        "standard": {
            "all": f"{EVAL_TABLES_BASE}/reward_logs_20251224_062618/Safe-RLHF-RM-hummer-all",
        },
        "qr": {
            "all": f"{EVAL_TABLES_BASE}/quantile_logs_20251224_151459/Safe-RLHF-RM-hummer-all",
        },
    },
}


# ============================================================
# 설정 가져오기 함수
# ============================================================

def get_dataset_config(dataset_name: str, auto_latest: bool = False) -> dict:
    """
    데이터셋 설정 가져오기
    
    Args:
        dataset_name: "pku", "helpsteer", "hummer", "hummer-all"
        auto_latest: True이면 최신 로그 폴더를 자동 탐색
    
    Returns:
        설정 딕셔너리
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {AVAILABLE_DATASETS}")
    
    dimensions = DATASET_DIMENSIONS[dataset_name]
    dataset_type = "hummer" if dataset_name.startswith("hummer") else "default"
    
    # 하위 디렉토리 설정
    std_subdir = "alpaca-8b-reproduced-llama-3"
    qr_subdir = "default"
    
    if auto_latest:
        # 자동 탐색
        standard_base_paths = {}
        qr_base_paths = {}
        
        for dim in dimensions:
            rm_name = get_dimension_rm_name(dim, dataset_type)
            
            std_path = find_latest_log_path("reward_logs", rm_name)
            qr_path = find_latest_log_path("quantile_logs", rm_name)
            
            if std_path:
                standard_base_paths[dim] = std_path
            if qr_path:
                qr_base_paths[dim] = qr_path
        
        print(f"[AUTO] Dataset: {dataset_name}")
        for dim in dimensions:
            print(f"  {dim}:")
            print(f"    Standard: {standard_base_paths.get(dim, 'NOT FOUND')}")
            print(f"    QR: {qr_base_paths.get(dim, 'NOT FOUND')}")
    else:
        # 하드코딩된 경로 사용
        standard_base_paths = HARDCODED_PATHS[dataset_name]["standard"]
        qr_base_paths = HARDCODED_PATHS[dataset_name]["qr"]
    
    # 전체 경로 생성
    standard_paths = {}
    qr_paths = {}
    
    for dim in dimensions:
        if dim in standard_base_paths:
            full_std = os.path.join(standard_base_paths[dim], std_subdir)
            standard_paths[dim] = full_std if os.path.exists(full_std) else standard_base_paths[dim]
        
        if dim in qr_base_paths:
            full_qr = os.path.join(qr_base_paths[dim], qr_subdir)
            qr_paths[dim] = full_qr if os.path.exists(full_qr) else qr_base_paths[dim]
    
    return {
        "name": DATASET_NAMES[dataset_name],
        "dimensions": dimensions,
        "standard_base_paths": standard_base_paths,
        "qr_base_paths": qr_base_paths,
        "standard_paths": standard_paths,
        "qr_paths": qr_paths,
        "dataset_paths": DATASET_PATHS.get(dataset_name, {}),
        "standard_subdir": std_subdir,
        "qr_subdir": qr_subdir,
    }


def get_conflict_dims(dataset_name: str) -> tuple:
    """
    Conflict 분석용 차원 쌍 반환 (KL/JS Divergence 계산용)
    
    Returns:
        (dim1, dim2) 또는 None
    """
    if dataset_name == "pku":
        return ("helpful", "safe")
    return None


def add_dataset_argument(parser: argparse.ArgumentParser):
    """ArgumentParser에 --dataset, --auto-latest 인자 추가"""
    parser.add_argument(
        "--dataset",
        type=str,
        choices=AVAILABLE_DATASETS,
        default="pku",
        help=f"Dataset to analyze: {AVAILABLE_DATASETS} (default: pku)"
    )
    parser.add_argument(
        "--auto-latest",
        action="store_true",
        help="Automatically find the latest log folders in eval_tables"
    )
    return parser


def get_output_dir(dataset_name: str, base_dir: str = None) -> str:
    """데이터셋별 출력 디렉토리 (experiment/standard_vs_qr/figures/ 내)"""
    if base_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(current_dir, "figures")
    output_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config test")
    add_dataset_argument(parser)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Auto-latest: {args.auto_latest}")
    print(f"{'='*60}")
    
    config = get_dataset_config(args.dataset, auto_latest=args.auto_latest)
    print(f"Name: {config['name']}")
    print(f"Dimensions: {config['dimensions']}")
    print(f"\nStandard Paths:")
    for dim, path in config['standard_paths'].items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {dim}: {exists} {path}")
    print(f"\nQR Paths:")
    for dim, path in config['qr_paths'].items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {dim}: {exists} {path}")
    print(f"\nConflict dims: {get_conflict_dims(args.dataset)}")
