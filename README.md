# EMFN 시간별 풍력발전량 예측

상명풍력 발전량과 새별오름 AWS 883 관측을 이용합니다. 메인 모델은 **v3 `src/models/emfn.py`**입니다. 기상의 **조건부 예측 기여**, 발전량의 유계 구간과 영발전 질량을 검증합니다. 문서 정리 기준일은 **2026-10-05**입니다.

## 현재 상태

구현·평가 오류 수정과 확률 베이스라인 추가에 이어 **현재 14개 베이스라인 + EMFN 전체를 새로 학습·평가**했습니다. 최신 실행은 `all_baselines_20261005T143556+0900`이며 15개 모델 계열, 6개 시계, 234개 모델·시계·시드 조합, 검증/평가 468행입니다. 모든 결과는 **과거에 모델 개발에 활용한 2025년 개발 벤치마크**이며 독립적인 블라인드 검증이 아닙니다.

| 단계 | 완료 범위 | 기록 |
|---|---|---|
| 오류 수정 후 재실험 | 10개 모델 × 6개 시계, seed 42; 구조 변형 4개 × 2개 시계 | [실험 이력](reports/EXPERIMENT_HISTORY.md), [기존 전체 표](reports/tables/comprehensive_extended_metric_matrix.csv) |
| 확률 베이스라인 확장 | 기존 10개 + ARIMA·24h LightGBM 3종 = 84개 평가 행 | [확장 보고서](reports/probabilistic_baselines.md) |
| 검증 CRPS·다중 시드 | EMFN 포함 6개 모델, 6개 시계, 시드 42·2026·3407; 96개 조합, 검증/평가 192행 | [다중 시드 보고서](reports/multiseed_crps_results.md) |
| 논문 보완 준비 | 2025 AWS 분→시간 재현, 기존 예측 층별 진단, 구조 대조군 구현·합성 학습 검증 | [보완 작업 기록](reports/research_readiness_20261005.md) |
| 전체 재실험·노트북 동기화 | 15개 모델, 3시드, 6시계; 234개 조합/468개 지표 행; A-to-Z 노트북 5개 | [최신 전체 보고서](reports/all_baselines_results.md), [노트북 안내](notebooks/README.md) |

최신 실행은 **신경망 126개와 LightGBM 모델 묶음 72개**, ARIMA 차수 후보 5개, Prophet 시드별 3개를 새로 적합했습니다. Persistence는 12개 시계 조합을 다시 계산했습니다. ARIMA·Persistence는 결정론 결과를 시드별로 복제하지 않으며 Prophet은 한 시드의 달력 예측을 여러 시계에서 평가합니다. 234는 독립적인 적합 호출 수가 아닌 모델·시계·시드 조합 수입니다. 신경망은 시드별 점수의 평균·표본 SD를 보고하며 시드 앙상블이 아닙니다.

EMFN의 6개 시계 평균 CRPS는 **1.4642 ± 0.0126**, 허들 분위수 LightGBM은 **1.4607**입니다. EMFN은 +1h/+3h 평균 CRPS가 가장 낮았고, 허들 LightGBM의 영발전 Brier는 6개 시계 모두 더 낮았습니다. 전반적인 확률 예측 우월성은 입증되지 않았습니다. 이전 실행의 시간 블록 구간을 이번 실행의 검정 결과로 재사용하지 않습니다.

GRU의 마지막 ReLU를 선형 출력으로 고쳐 음수 초기 출력의 기울기 차단을 제거했습니다. 최신 실행의 검증 상수 예측 경고는 0개입니다. Prophet은 train-only 달력·추세 모델이며 최근 발전량을 매 시점 반영하는 모델들과 정보 사용이 다릅니다. 장기 lag LightGBM도 검증 CRPS로 선택하고 NaN 자체 처리를 적용했습니다. [설정과 변경 사항](reports/all_baselines_results.md)

최종 확인: **44개 회귀 테스트, 노트북 5개 전체 실행, 468개 지표 행 재계산·체크포인트 재로딩 검증을 통과**했습니다. 원자료 전처리 재구축도 기존 데이터와 값·SHA-256이 일치합니다. [통합 검증 기록](reports/research_workflow_validation.json)

## 현재 논문 구성안

**실험한 모델 목록과 본문에 배치할 모델 목록을 구분합니다.** 본문은 표준 모델 중심으로 간추리고, 이미 수행한 확률 비교는 부록·보조 실험에 보존합니다. CRPS 주 지표와 영발전 Brier 핵심 보조 지표는 유지합니다.

- 현재 본문 표: **EMFN, Naive Persistence, ARIMA, LSTM, 일반 LightGBM(24h)**의 총 5개 모델. 표는 +1h/+6h/+24h, 곡선은 6개 시계 전체를 사용합니다.
- 보조 확률 표: **EMFN, 분위수 LightGBM, 허들 분위수 LightGBM**. 본문에서도 해당 비교의 요점과 한계를 요약합니다.
- CNN-LSTM·TCN·GRU·Transformer·DLinear·Prophet·Diurnal·장기 lag LightGBM도 최신 전체 결과에 보존합니다. 다른 실행의 점수를 본문에 임의로 합치지 않습니다.

[본문용 전체 시계 표](reports/tables/paper_main_standard_baselines.csv), [대표 시계 축약 표](reports/tables/paper_main_compact.csv), [보조 확률 표](reports/tables/paper_supplementary_probability.csv)는 [표시 설정](configs/paper_presentation.yaml)에 따라 최신 검증 결과의 행만 추출합니다. 표시 단계에서는 재학습·선택·점수 수정을 하지 않습니다. `paper_five_comparators.csv`는 이전 단일 시드 구성안이며 최신 본문 표가 아닙니다.

## 평가 규약과 열 의미

- 주 지표 **CRPS**, 핵심 보조 **영발전 Brier**, 추가 지표 **RMSE·MAE·영발전 AUPRC**를 사용합니다. 확률 모델은 RMSE에 평균, MAE에 중앙값을 사용합니다.
- 결정론 모델은 자체 점출력을 [0,21]로 clip합니다. Dirac CRPS는 MAE와 같고, 확률 출력이 없으면 Brier/AUPRC는 N/A입니다. 이들만으로 확률 예측 우수성을 입증하지 않습니다.
- ARIMA는 가우시안 예측분포를 [0,21]로 censor한 기준선입니다. 분위수 LightGBM은 정렬·범위 제한·끝점 보간을 사용합니다. 이 분포들의 평균은 단순히 raw 평균을 clip한 값과 다를 수 있습니다. [구성 상세](reports/probabilistic_baselines.md)
- 최신 학습 선택은 검증 CRPS입니다. EMFN 학습률 스케줄러도 CRPS를 관찰하지만 학습 손실은 기존 composite loss, 다른 신경망은 Huber를 유지합니다. Persistence와 고정 설정 Prophet에는 선택이 없습니다.
- 시드 SD는 학습 초기화의 변동, 24h/168h paired 블록 구간은 고정된 학습 실행을 조건으로 한 평가 시계열 변동입니다. 다중 비교·계절 비정상성·독립 검증 문제를 해결한 구간은 아닙니다.

| 결과 파일 | `mae`의 의미 | 선택/실행 범위 |
|---|---|---|
| `comprehensive_extended_metric_matrix.csv` | EMFN 평균의 MAE; 중앙값은 `mae_median` | 기존 10개 모델, 신경망 val loss, seed 42 |
| `probabilistic_extended_benchmark.csv` | 확률 모델 중앙값; 평균은 `mae_mean` | 기존 신경망 재사용 + 신규 모델 val CRPS |
| `paper_five_comparators.csv` | 위 확장 표의 중앙값 MAE를 발췌; `mae_mean` 열 없음 | 이전 본문 구성안 |
| `multiseed_crps_metrics.csv` | 확률 모델 중앙값; 평균은 `mae_mean` | 통일된 val CRPS, 검증/평가 및 시드별 기록 |
| `multiseed_crps_summary.csv`, 새 본문/보조 표 | `mae_mean`은 **중앙값 MAE의 시드 평균**; `mae_sd`는 시드 SD | 요약 열의 `_mean`은 분포 평균을 뜻하지 않음 |
| `all_baselines_metrics.csv` | 확률 모델 중앙값의 MAE; `mae_mean`은 분포 평균 예측의 MAE | 최신 15개 모델, 468행 |
| `all_baselines_summary.csv` | `_mean`/`_sd`는 시드 집계; `mae_mean`은 중앙값 MAE의 시드 평균 | 최신 180행 요약 |

## 구조와 데이터

공유 TCN에서 magnitude/zero adapter로 나뉘고 HF/AR 경로와 선택적 기상 특징 MLP를 사용합니다. `sigmoid(z_zero)`는 변수명과 달리 **p=P(Y>0|X)**입니다. 양수 조건부 분포는 `Y/21 ~ Beta(alpha,beta)`이고, 전체 평균은 `21*p*alpha/(alpha+beta)`입니다. 0에 질량이 있지만 21에 별도 점질량은 없습니다.

유계 구간과 영발전 질량을 반영한 통계 모델이며 제조사의 시동·정지 제어 법칙이나 기상 변수의 인과 효과를 식별하지 않습니다. GroupNorm은 입력 윈도우 안의 시간 위치를 함께 정규화하므로 모든 중간 토큰의 엄격한 prefix causality도 주장하지 않습니다.

학습은 2023-01-01 01:00~2024-06-30 23:00, 검증은 2024년 하반기, 평가는 2025년입니다. 발전량은 MWh, 상한은 21 MWh, 영발전은 정확한 관측 0입니다. 기상 표준화는 학습 구간에만 적합합니다. 미래 참조 없이 최대 3시간 ffill하고, 남은 결측 윈도우는 시간축을 압축하지 않고 제외합니다. 관측이 시간 구간 종료 즉시 가용하다는 오프라인 가정의 실제 타당성은 미확인입니다.

시퀀스 모델과 24h LightGBM은 발전량+기상 6종의 24시간 입력입니다. **LightGBM(+Weather)은 최대 168h lag·달력 특징을 사용한 별도 기준선**이며 이번에도 재학습했습니다. ARIMA는 발전량 과거 상태를 누적하고 Prophet은 train-only 달력·추세 예측입니다. 공통 평가 표본 수는 시계 순서 [1,3,6,9,12,24]h에서 [8736,8734,8731,8728,8725,8713]입니다.

전체 파일은 28,440행 중 유효 발전량 28,368개이며 2026년 일부도 포함합니다. 2023~2025는 26,303개, 정확한 0은 4,799개(18.25%)입니다. 최소 양수는 0.024316 MWh입니다. [데이터 QA](reports/tables/data_quality_by_scope.csv)

공식 단위·집계 시각·계량 경계, 배포 지연, 정격·정비·출력제어 이력은 확인이 필요합니다. 공식 설명은 송전량으로 기술하지만 본문 kWh와 컬럼 정의 MWh가 상충하며, 사용자가 별도 설명을 받은 이력은 없습니다. 2026 AWS 분 자료는 존재합니다. 2025년 자료에서 변환을 검증한 2026년 시간 기상 후보를 별도 보관했으며 기존 병합 데이터에는 반영하지 않았습니다. 2026년 예측 성능은 평가하지 않았습니다. 풍속 조건부 오차나 임의 비대칭 오차를 실제 cut-in 검증·시장 수익으로 해석하지 않습니다.

## 환경 설정 및 재현 안내

### 1. 로컬 환경 설정 (Conda 권장)

저장소 루트에서 `environment.yml` 하나로 검증된 환경을 구축할 수 있습니다:

```bash
# 1) Conda 가상환경 생성 및 활성화
conda env create -f environment.yml
conda activate wind_power

# 2) Jupyter 노트북 커널 등록 (노트북 워크플로 실행 시 필요)
python -m ipykernel install --user --name wind_power --display-name "Python (wind_power)"

# 3) (선택) Windows NVIDIA GPU 가속이 필요한 경우
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

> **pip 단독 설치:** Python 3.11 가상환경에서 `pip install -r requirements.txt`로도 설치 가능합니다. `requirements.txt`에 명시된 버전 조합(NumPy 2.2.6, SciPy 1.15.3, statsmodels 0.14.6)은 Windows 환경에서의 ARIMA 상태공간 바이너리 호환성을 보장합니다.

### 2. Google Colab에서 실행하기

로컬 GPU 사양이 부족하거나 웹 브라우저에서 바로 실행하려는 경우, 아래 배지를 클릭하여 Colab 런타임에서 저장소를 클론하고 결과를 열람하거나 모델을 실행할 수 있습니다:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-axis/Wind-Power-Forecasting/blob/main/colab/wind_power_colab.ipynb)

Colab 노트북(`colab/wind_power_colab.ipynb`)은 다음과 같이 동작합니다:
1. 저장소를 클론하고 `colab/requirements-colab.txt`로 Colab 사전 설치 스택(CUDA PyTorch 등)과 충돌 없이 필요한 패키지만 빠르게 추가합니다.
2. 커밋된 `reports/` 및 `data/`를 바탕으로 학습 없이 즉시 종합 벤치마크 결과와 시각화를 조회합니다.
3. GPU 런타임(T4 등)을 활성화하면 Colab 인스턴스 내에서 직접 재학습(`RUN_EMFN_EXAMPLE=True` 등)도 가능합니다.

### 3. 검증 및 실험 실행 명령

저장소 루트에서:

```bash
# 단위 테스트 실행 (로컬 test/ 디렉터리 내 44개 테스트)
python -m unittest discover -s test -p "test_*.py" -v

# 검증된 최신 실행으로 현재 본문/부록 표만 추출 (학습 없음)
python -m src.experiments.export_paper_views

# 새 전체 실험이 필요한 경우에만 실행
python -m src.experiments.run_all_baselines
python -m src.experiments.verify_all_baselines --run models/runs/<new_run_id>
python -m src.experiments.report_all_baselines
python -m src.experiments.export_paper_views
python -m src.workflows.execute_notebooks
```

과거 재현용 `run_comprehensive_extended_matrix`와 `run_comprehensive_ablation_matrix`는 명시적으로 loss 선택 규약을 유지합니다. 전자는 전처리 데이터와 이전 결과 표도 갱신하므로 단순 보고서 열람용 명령이 아닙니다. 과거와 현재 실험을 섞어 선택 효과만의 어블레이션으로 해석하지 않습니다.

- 최신 모델/예측/epoch 이력: [실행 manifest](reports/all_baselines_manifest.json)가 가리키는 `models/runs/<run_id>/`. 기존 `models/checkpoints/emfn_*.pt`는 이전 단일 시드 가중치입니다.
- 실행 당시 코드·설정은 run의 `source.zip`과 해시로 보존합니다. 현재 파일은 문서/코드 정합성 수정으로 달라질 수 있습니다. 중단 실행 재개는 당시 코드·설정이 동일할 때만 가능합니다.
- [실험 이력](reports/EXPERIMENT_HISTORY.md), [남은 과제](reports/baseline_benchmarks_and_future_plan.md), [v1/v2/v3 구조](src/models/history/README.md).
- [노트북 5개](notebooks/README.md)는 원자료·전처리 → 입력 EDA → 베이스라인 → EMFN → 전체 비교를 재현합니다. 셀은 `src/workflows/research.py`를 호출하며 상세 모델·학습 코드는 `.py`에 있습니다. 기본값은 저장 결과 열람이고 명시적 스위치에서만 재학습합니다. 기존 4개 노트북은 로컬 아카이브에 보존했습니다.
- 로컬 아카이브·run 본체는 Git에서 제외하며 [수정 전 해시 인덱스](reports/archive/pre_correction_manifest.json)는 추적 대상으로 유지합니다. 문서 위치 이동은 결과 삭제나 평가 독립성 확보를 뜻하지 않습니다.
