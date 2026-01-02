
PKU-Alignment - [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf)

1. **Multi-Objective Alignment**: 단일 목표가 아닌 다중 목표(helpful, safe, verbosity 등)를 동시에 최적화
2. **Pareto Optimality**: Panacea를 통해 다양한 선호도 조합에서 최적의 모델 제공
3. **Uncertainty-Aware Reward**: Quantile로 보상의 불확실성 추정


| 기능 | Safe RLHF (원본) | Safe-RLHF-HK (확장) |
|------|------------------|---------------------|
| 목표 | 2개 (reward, cost) | N개 (동적 설정 가능) |
| 알고리즘 | PPO, PPO-Lag | + Panacea PPO, PG-Panacea |
| Reward Model | Standard (기대값) | + Quantile |
| 선호도 적응 | 고정 | SVD LoRA로 실시간 적응 |
| 평가 메트릭 | 기본 | + Pareto 메트릭 (HV, IP, SP, Spacing) |

---

## 디렉토리 구조

```
safe-rlhf-hk/
├── safe_rlhf/              # 핵심 라이브러리
│   ├── algorithms/         # 학습 알고리즘
│   │   ├── ppo/            # 기본 PPO
│   │   ├── ppo_lag/        # Lagrangian PPO
│   │   ├── ppo_panacea/    # Panacea PPO (Multi-Objective)
│   │   ├── pg_panacea/     # Pure Policy Gradient Panacea
│   │   └── dpo/            # Direct Preference Optimization
│   ├── models/             # 모델 정의
│   │   ├── panacea.py      # Panacea SVD-LoRA 모델
│   │   ├── pretrained.py   # 사전학습 모델 로더
│   │   └── score_model/    # 아키텍처별 Score 모델
│   ├── values/             # 보상/비용 모델
│   │   ├── reward/         # Standard Reward Model
│   │   ├── reward_qr/      # Quantile RM
│   │   ├── cost/           # Standard Cost Model(legacy)
│   │   └── cost_qr/        # Quantile Cost(legacy)
│   ├── trainers/           # 학습 트레이너
│   │   ├── base.py         # 기본 트레이너
│   │   └── rl_trainer.py   # RL 트레이너
│   ├── datasets/           # 데이터셋 로더
│   └── configs/            # DeepSpeed 설정
│
├── scripts/                # 학습 실행 스크립트 (Shell)
│   ├── ppo-panacea-*.sh    # Panacea PPO 학습
│   ├── pg-panacea-*.sh     # PG Panacea 학습
│   ├── reward-model-*.sh   # Reward Model 학습
│   ├── sft-*.sh            # SFT 학습
│   └── run_all_*.sh        # 일괄 실행 스크립트
│
├── experiment/             # 실험 및 분석 코드
│   ├── panacea/            # Panacea 관련 실험
│   │   ├── figures/                        # Pareto Front 시각화 결과
│   │   ├── benchmark_paper_metrics.py      # Standard RM 메트릭
│   │   ├── benchmark_paper_metrics_qr.py   # QR RM 메트릭
│   │   └── panacea_vector_imminsik.py      # Pareto Front 시각화
│   ├── standard_vs_qr/     # Standard vs QR 비교 분석
│   │   ├── figures/                        # 비교 분석 시각화 결과
│   │   ├── config.py                       # 데이터셋별 설정
│   │   └── run_all_analysis.py             # 전체 분석 파이프라인
│   └── old/                # 이전 실험 코드
│
├── tools/                  # 유틸리티 스크립트
│   ├── data_processing/    # 데이터 변환/분할
│   ├── upload/             # HuggingFace 업로드
│   └── test/               # 모델 테스트
│
├── datasets/               # 데이터셋 모음
│   ├── pku_saferlhf/       # PKU-SafeRLHF (helpful, safe)
│   ├── hummer/             # Hummer 6차원 데이터셋
│   ├── hummer_all/         # Hummer 전체
│   ├── preference/         # HelpSteer 5차원
│   └── helpsteer/          # HelpSteer annotator별
│
├── output/                 # 학습된 모델 체크포인트
│   ├── rm_*/               # Reward Model 출력
│   ├── rm_qr_*/            # Quantile RM 출력
│   └── ppo-panacea-*/      # Panacea PPO 출력
│
└── eval_tables/            # 평가 결과 테이블 (scalar, qr reward value - eval 데이터)
```

---

## 구성 요소

### 1. Panacea (Multi-Objective RLHF)

**Panacea**는 다중 목표 최적화를 위한 RLHF 프레임워크입니다. SVD 기반 LoRA를 사용하여 다양한 선호도 벡터에 대해 실시간으로 모델을 적응시킵니다.

#### 핵심 개념

- **SVD-based LoRA**: 선호도 벡터를 singular value로 주입
- **Pareto Front**: 다중 목표 간 trade-off 최적화
- **Linear Scalarization / Tchebycheff**: 목표 aggregation 방법

#### 관련 파일

| 파일 | 역할 |
|------|------|
| `safe_rlhf/algorithms/ppo_panacea/trainer.py` | Panacea PPO 학습 로직 |
| `safe_rlhf/algorithms/ppo_panacea/helpers/` | 손실 계산, 메트릭, 선호도 관리 |
| `safe_rlhf/models/panacea.py` | PanaceaModelForCausalLM (SVD-LoRA) |

#### 실행 스크립트

```bash
# Helpful/Safe 2차원 Panacea (Standard RM)
bash scripts/ppo-panacea-HH-standard.sh

# Helpful/Safe 2차원 Panacea (Quantile RM)
bash scripts/ppo-panacea-HH-qr.sh

# Hummer 6차원 Panacea
bash scripts/ppo-panacea-hummer-qr.sh
```

#### 주요 파라미터

**Panacea 파라미터**
```bash
PANACEA_LORA_DIM=8              # LoRA rank
PANACEA_PREF_DIM=2              # 선호도 차원 수 (목표 개수, 자동 설정)
PANACEA_SCALING=512             # SVD scaling factor
PANACEA_AGGREGATION="linear_scalarization"  # 또는 "tchebycheff"
```

**PPO 학습 추가설정 **
```bash
# Quantization (메모리 최적화)
QUANTIZE_REWARD_MODELS=True     # Reward 모델 4-bit 로드
QUANTIZE_REFERENCE_MODEL=True   # Reference 모델 4-bit 로드

# Save
SAVE_16BIT=True                 # bf16으로 저장
```

---

### 2. Quantile Reward Model

**Quantile RM**은 보상의 분포를 학습하여 불확실성을 추정합니다. 기존 기대값 기반 RM과 달리, 여러 분위수(quantile)에서의 보상 값을 예측합니다.

#### 핵심 개념

- **분위수 예측**: q0.1, q0.2, ..., q0.9 등 다중 분위수 출력
- **불확실성 추정**: 분위수 간 차이로 불확실성 측정

#### 관련 파일

| 파일 | 역할 |
|------|------|
| `safe_rlhf/values/reward_qr/trainer.py` | QR Reward Model 학습 |
| `safe_rlhf/values/cost_qr/trainer.py` | QR Cost Model 학습 |

#### 실행 스크립트

```bash
# Quantile Reward Model 학습
bash scripts/reward-model-dimension-HH-qr.sh

# 전체 차원 QR RM 일괄 학습
bash scripts/run_all_reward_models_qr_HH.sh
```

---

### 3. Standard Reward Model

기존의 기대값 기반 보상 모델입니다. 단일 스칼라 보상 값을 예측합니다.

#### 관련 파일

| 파일 | 역할 |
|------|------|
| `safe_rlhf/values/reward/trainer.py` | Standard RM 학습 |
| `safe_rlhf/values/cost/trainer.py` | Standard Cost Model 학습 |

#### 실행 스크립트

```bash
# Standard Reward Model 학습
bash scripts/reward-model-dimension-HH.sh

# 전체 차원 Standard RM 일괄 학습
bash scripts/run_all_reward_models_HH.sh
```

---

## 데이터셋

### 데이터셋 구조

| 디렉토리 | 설명 | 차원 |
|----------|------|------|
| `datasets/pku_saferlhf/` | PKU-SafeRLHF | helpful, safe |
| `datasets/hummer/` | Hummer 데이터셋 | accuracy, conciseness, depth, empathy, specificity, tone |
| `datasets/preference/` | HelpSteer 파생 | verbosity, complexity, helpfulness, correctness, coherence |
| `datasets/helpsteer/` | HelpSteer 원본 (annotator별) | 위와 동일 |

### 데이터 포맷

```json
{
  "prompt": "사용자 질문...",
  "response_0": "응답 A...",
  "response_1": "응답 B...",
  "better_response_id": 0
}
```

---

### 1단계: SFT (Supervised Fine-Tuning)

```bash
bash scripts/sft-deepspeed.sh
```

### 2단계: Reward Model 학습

```bash
# Standard RM
bash scripts/reward-model-dimension-HH.sh

# Quantile RM
bash scripts/reward-model-dimension-HH-qr.sh
```

### 3단계: PPO/Panacea 학습

```bash
# Standard RM 사용
bash scripts/ppo-panacea-HH-standard.sh

# Quantile RM 사용
bash scripts/ppo-panacea-HH-qr.sh
```

### 4단계: 평가 및 시각화

```bash
# Pareto Front 시각화
python experiment/panacea/panacea_vector_imminsik.py

# 벤치마크 메트릭 계산
python experiment/panacea/benchmark_paper_metrics_qr.py
```

---

## 주요 스크립트

### PPO/Panacea 학습 스크립트

현재 모든 스크립트는 `PKU-Alignment/alpaca-8b-reproduced-llama-3` 베이스 모델을 사용합니다.

| 스크립트 | 설명 | 차원 | RM 유형 |
|----------|------|------|---------|
| `ppo-panacea-HH-standard.sh` | PKU-SafeRLHF (helpful, safe) | 2D | Standard |
| `ppo-panacea-HH-qr.sh` | PKU-SafeRLHF (helpful, safe) | 2D | Quantile |
| `ppo-panacea-hummer-standard.sh` | Hummer (accuracy, conciseness, depth, empathy, specificity, tone) | 6D | Standard |
| `ppo-panacea-hummer-qr.sh` | Hummer (accuracy, conciseness, depth, empathy, specificity, tone) | 6D | Quantile |
| `ppo-panacea-standard.sh` | HelpSteer (verbosity, complexity, helpfulness, correctness, coherence) | 5D | Standard |
| `ppo-panacea-qr.sh` | HelpSteer (verbosity, complexity, helpfulness, correctness, coherence) | 5D | Quantile |

### Reward Model 학습 스크립트

| 스크립트 | 설명 |
|----------|------|
| `reward-model-dimension-HH.sh` | PKU-SafeRLHF Standard RM |
| `reward-model-dimension-HH-qr.sh` | PKU-SafeRLHF Quantile RM |
| `reward-model-dimension-hummer.sh` | Hummer Standard RM |
| `reward-model-dimension-qr-hummer.sh` | Hummer Quantile RM |
| `run_all_reward_models_*.sh` | 전체 차원 일괄 학습 |

### 기타 스크립트

| 스크립트 | 설명 |
|----------|------|
| `sft-deepspeed.sh` | DeepSpeed SFT 학습 |
| `pg-panacea-*.sh` | Pure Policy Gradient Panacea |
| `convert_to_bf16.py` | 모델 bf16 변환 |

---

## 실험 및 분석

### Panacea 실험

| 파일 | 역할 |
|------|------|
| `experiment/panacea/benchmark_paper_metrics.py` | Standard RM 기반 Panacea 메트릭 계산 |
| `experiment/panacea/benchmark_paper_metrics_qr.py` | Quantile RM 기반 Panacea 메트릭 계산 |
| `experiment/panacea/panacea_vector_imminsik.py` | Pareto Front 2D/3D 시각화 |
| `experiment/panacea/panacea_infer_standard.py` | Panacea 모델 추론 |

### 평가 메트릭 (panacea)

| 메트릭 | 설명 |
|--------|------|
| **Hypervolume (HV)** | Pareto Front가 지배하는 공간의 부피 |
| **Inner Product (IP)** | 선호도 벡터와 평가 벡터의 내적 평균 |
| **Sparsity (SP)** | Pareto Front 점들의 균등 분포 정도 |
| **Spacing** | 최근접 이웃 거리의 표준편차 |

### Standard vs QR 비교 (reward model value 비교)

모든 분석 스크립트는 `--dataset` 옵션으로 데이터셋을 선택할 수 있습니다.

| 데이터셋 | 차원 | 옵션 |
|----------|------|------|
| PKU-SafeRLHF | 2D (helpful, safe) | `--dataset pku` |
| HelpSteer | 5D (helpfulness, coherence, complexity, correctness, verbosity) | `--dataset helpsteer` |
| Hummer | 6D 개별 차원 (accuracy, conciseness, depth, empathy, specificity, tone) | `--dataset hummer` |
| Hummer-All | 6D 통합 단일 모델 | `--dataset hummer-all` |

| 파일 | 역할 |
|------|------|
| `experiment/standard_vs_qr/config.py` | 공통 설정 (데이터셋별 경로/차원 관리) |
| `experiment/standard_vs_qr/run_all_analysis.py` | 전체 분석 파이프라인 자동 실행 |
| `experiment/standard_vs_qr/compare_qr_vs_standard_histogram.py` | 히스토그램 비교 |
| `experiment/standard_vs_qr/compare_qr_vs_standard_kde.py` | KDE 비교 |
| `experiment/standard_vs_qr/comprehensive_roc_analysis.py` | ROC 분석 |
| `experiment/standard_vs_qr/comprehensive_statistical_analysis.py` | 통계 분석 (AUC, Cohen's d 등) |
| `experiment/standard_vs_qr/hard_case_analysis.py` | High Spread (불확실성) 분석 |
| `experiment/standard_vs_qr/conflict_analysis.py` | 충돌 분석 (PKU only: helpful vs safe) |

---

## 유틸리티 도구

### tools/data_processing/

| 파일 | 역할 |
|------|------|
| `convert_hummer_split.py` | Hummer 데이터셋 train/test 분할 |
| `convert_disagreements_to_preference.py` | 불일치 데이터 → 선호도 데이터 변환 |
| `merge_datasets.py` | 여러 annotator 데이터 병합 |
| `split_all_dimensions.py` | 전체 차원 데이터 분할 |
| `convert_qr_models.py` | QR 모델 변환 |

### tools/upload/

| 파일 | 역할 |
|------|------|
| `upload_ppo_to_hf.py` | PPO 모델 HuggingFace 업로드 (HH, helpsteer, hummer 지원) |
| `upload_rm_to_hf.py` | Reward Model HuggingFace 업로드 (standard, qr 지원) |

> **Note**: 스크립트는 프로젝트 루트를 자동으로 찾아 어디서든 실행 가능합니다.

### tools/test/

| 파일 | 역할 |
|------|------|
| `test_coherence_rm.py` | Coherence RM 테스트 |
| `test_sft_model.py` | SFT 모델 테스트 |

---

## 환경 설정 

### 1. Conda 환경 활성화

```bash
conda activate hk
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 프로젝트 설치

```bash
pip install -e .
```

---

## 빠른 시작 가이드

### 전체 파이프라인 (PKU-SafeRLHF 2차원: helpful, safe)

```bash
# 1. 환경 활성화
conda activate hk
cd /home/hail/safe-rlhf-hk

# 2. Reward Model 학습 (Standard)
bash scripts/reward-model-dimension-HH.sh

# 3. Reward Model 학습 (Quantile)
bash scripts/reward-model-dimension-HH-qr.sh

# 4. Panacea PPO 학습 (Standard RM 사용)
bash scripts/ppo-panacea-HH-standard.sh

# 5. Panacea PPO 학습 (Quantile RM 사용)
bash scripts/ppo-panacea-HH-qr.sh

# 6. 평가 및 메트릭 계산
python experiment/panacea/benchmark_paper_metrics.py
python experiment/panacea/benchmark_paper_metrics_qr.py

# 7. Pareto Front 시각화
python experiment/panacea/panacea_vector_imminsik.py
```

---

## 상세 실행 방법

```bash
export WANDB_API_KEY='wandb-key'
```

### 1. Reward Model 학습

#### Standard Reward Model

```bash
# PKU-SafeRLHF (helpful, safe)
bash scripts/reward-model-dimension-HH.sh

# 전체 차원 일괄 학습
bash scripts/run_all_reward_models_HH.sh

# Hummer 6차원
bash scripts/reward-model-dimension-hummer.sh

# HelpSteer 5차원
bash scripts/reward-model-dimension.sh
```

#### Quantile Reward Model

```bash
# PKU-SafeRLHF (helpful, safe)
bash scripts/reward-model-dimension-HH-qr.sh

# 전체 차원 일괄 학습
bash scripts/run_all_reward_models_qr_HH.sh

# Hummer 6차원
bash scripts/reward-model-dimension-qr-hummer.sh

# HelpSteer 5차원
bash scripts/reward-model-dimension-qr.sh
```

### 2. Panacea PPO 학습

#### 2차원 (helpful, safe)

```bash
# Standard RM 기반
bash scripts/ppo-panacea-HH-standard.sh

# Quantile RM 기반
bash scripts/ppo-panacea-HH-qr.sh
```

#### 6차원 (Hummer)

```bash
# Standard RM 기반
bash scripts/ppo-panacea-hummer-standard.sh

# Quantile RM 기반
bash scripts/ppo-panacea-hummer-qr.sh
```

#### 5차원 (HelpSteer)

```bash
# Standard RM 기반
bash scripts/ppo-panacea-standard.sh

# Quantile RM 기반
bash scripts/ppo-panacea-qr.sh
```

### 3. Pure Policy Gradient (PG) Panacea

```bash
# Standard RM 기반
bash scripts/pg-panacea-standard.sh

# Quantile RM 기반
bash scripts/pg-panacea-qr.sh
```

### 4. 평가 및 시각화

#### Pareto 메트릭 계산 (HV, IP, SP, Spacing)

```bash
# Standard RM 기반 Panacea
python experiment/panacea/benchmark_paper_metrics.py

# Quantile RM 기반 Panacea
python experiment/panacea/benchmark_paper_metrics_qr.py
```

#### Pareto Front 시각화

```bash
# 2D/3D Pareto Front 그래프 생성
python experiment/panacea/panacea_vector_imminsik.py
```

#### Standard vs Quantile 비교 분석

```bash
# 전체 분석 파이프라인 (단일 데이터셋)
python experiment/standard_vs_qr/run_all_analysis.py --dataset pku
python experiment/standard_vs_qr/run_all_analysis.py --dataset helpsteer
python experiment/standard_vs_qr/run_all_analysis.py --dataset hummer
python experiment/standard_vs_qr/run_all_analysis.py --dataset hummer-all

# 모든 데이터셋 일괄 분석
python experiment/standard_vs_qr/run_all_analysis.py --all

# 최신 로그 폴더 자동 탐색 (--auto-latest)
# eval_tables에서 reward_logs_*, quantile_logs_*를 각각 독립적으로 최신 버전 탐색
python experiment/standard_vs_qr/run_all_analysis.py --dataset hummer --auto-latest
python experiment/standard_vs_qr/run_all_analysis.py --all --auto-latest

# 개별 분석 스크립트 (데이터셋 지정)
python experiment/standard_vs_qr/compare_qr_vs_standard_histogram.py --dataset pku
python experiment/standard_vs_qr/compare_qr_vs_standard_kde.py --dataset helpsteer
python experiment/standard_vs_qr/comprehensive_roc_analysis.py --dataset hummer
python experiment/standard_vs_qr/comprehensive_statistical_analysis.py --dataset pku
python experiment/standard_vs_qr/hard_case_analysis.py --dataset hummer

# 개별 스크립트에서도 --auto-latest 사용 가능
python experiment/standard_vs_qr/comprehensive_roc_analysis.py --dataset hummer --auto-latest
python experiment/standard_vs_qr/hard_case_analysis.py --dataset hummer --auto-latest

# 충돌 분석 (PKU only: helpful vs safe)
python experiment/standard_vs_qr/conflict_analysis.py --dataset pku
```

> **`--auto-latest` 옵션**: `eval_tables/kms_test/`에서 `reward_logs_YYYYMMDD_HHMMSS`와 `quantile_logs_YYYYMMDD_HHMMSS`를 각각 독립적으로 최신 버전을 탐색합니다. 하드코딩된 경로 대신 동적으로 최신 평가 결과를 사용합니다.

분석 결과는 `experiment/standard_vs_qr/figures/{dataset}/` 폴더에 저장됩니다.

### 5. 모델 추론

```bash
# Panacea 모델 추론 (선호도 벡터 지정)
python experiment/panacea/panacea_inference.py

# Standard RM 기반 추론
python experiment/panacea/panacea_infer_standard.py
```

### 6. 모델 업로드 (HuggingFace)

```bash
# HF 토큰 설정
export HF_TOKEN='your_token_here'

# Reward Model 업로드
python tools/upload/upload_rm_to_hf.py --dimension helpful --type standard
python tools/upload/upload_rm_to_hf.py --all --type both

# PPO 모델 업로드
python tools/upload/upload_ppo_to_hf.py --type standard --dataset HH
python tools/upload/upload_ppo_to_hf.py --type both --dataset all
```

### 7. 유틸리티

#### 모델 bf16 변환

```bash
# 변환 대상 확인
python scripts/convert_to_bf16.py --status

# 전체 변환 (dry-run)
python scripts/convert_to_bf16.py --all --dry-run

# 실제 변환
python scripts/convert_to_bf16.py --all
```

#### 데이터 처리

```bash
# Hummer 데이터 분할
python tools/data_processing/convert_hummer_split.py

# HelpSteer 데이터 분할
python tools/data_processing/split_all_dimensions.py

# 데이터셋 병합
python tools/data_processing/merge_datasets.py
```

---

## 출력 결과

### 모델 체크포인트

| 디렉토리 | 내용 |
|----------|------|
| `output/rm_helpful_*/` | Helpful Standard RM |
| `output/rm_safe_*/` | Safe Standard RM |
| `output/rm_qr_helpful_*/` | Helpful Quantile RM |
| `output/rm_qr_safe_*/` | Safe Quantile RM |
| `output/ppo-panacea-HH-standard/` | Standard RM 기반 Panacea 모델 |
| `output/ppo-panacea-HH-qr/` | Quantile RM 기반 Panacea 모델 |

### 시각화 결과

각 실험 폴더 내 `figures/` 디렉토리에서 결과를 관리합니다.

| 디렉토리 | 내용 |
|----------|------|
| `experiment/panacea/figures/` | Pareto Front 2D/3D 그래프 |
| `experiment/standard_vs_qr/figures/pku/` | PKU-SafeRLHF 비교 분석 결과 |
| `experiment/standard_vs_qr/figures/helpsteer/` | HelpSteer 비교 분석 결과 |
| `experiment/standard_vs_qr/figures/hummer/` | Hummer 6D 개별 차원 분석 결과 |
| `experiment/standard_vs_qr/figures/hummer-all/` | Hummer 6D 통합 모델 분석 결과 |

### 평가 결과

| 디렉토리 | 내용 |
|----------|------|
| `eval_tables/kms_test/reward_logs_*/` | Standard RM 평가 결과 (자동 생성) |
| `eval_tables/kms_test/quantile_logs_*/` | Quantile RM 평가 결과 (자동 생성) |

> **Note**: 평가 로그는 Reward Model 학습 시 eval 단계에서 자동으로 생성됩니다. 폴더명에 타임스탬프(`YYYYMMDD_HHMMSS`)가 포함됩니다.

---

## 참고

1. **Panacea**: [Panacea: Pareto Alignment via Preference Adaptation for LLMs](https://proceedings.neurips.cc/paper_files/paper/2024/file/89f39d0b3d49a47606a165eefba2778c-Paper-Conference.pdf) (NeurIPS 2024)
2. **Safe RLHF**: [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2310.12773) (ICLR 2024 Spotlight)
3. **PKU-SafeRLHF**: [PKU-Alignment/PKU-SafeRLHF Dataset](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)

---

## 라이센스

이 프로젝트는 Apache License 2.0을 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

