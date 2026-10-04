# EMFN 시간별 풍력발전량 예측

상명풍력 발전량과 새별오름 AWS 883 관측을 이용합니다. 메인 모델은 **v3 `src/models/emfn.py`**입니다. 기상의 **조건부 예측 기여**, 발전량의 유계 구간과 영발전 질량을 검증합니다. 문서 정리 기준일은 **2026-10-05**입니다.

## 현재 상태

구현·평가 오류 수정, 확률 베이스라인 추가, 검증 CRPS 선택 기준 통일 및 3개 시드 비교를 완료했습니다. 모든 결과는 **과거에 모델 개발에 활용한 2025년 개발 벤치마크**이며 독립적인 블라인드 검증이 아닙니다.

| 단계 | 완료 범위 | 기록 |
|---|---|---|
| 오류 수정 후 재실험 | 10개 모델 × 6개 시계, seed 42; 구조 변형 4개 × 2개 시계 | [실험 이력](reports/EXPERIMENT_HISTORY.md), [기존 전체 표](reports/tables/comprehensive_extended_metric_matrix.csv) |
| 확률 베이스라인 확장 | 기존 10개 + ARIMA·24h LightGBM 3종 = 84개 평가 행 | [확장 보고서](reports/probabilistic_baselines.md) |
| 검증 CRPS·다중 시드 | EMFN 포함 6개 모델, 6개 시계, 시드 42·2026·3407; 96개 조합, 검증/평가 192행 | [다중 시드 보고서](reports/multiseed_crps_results.md) |

다중 시드 실행은 신경망 36개와 LightGBM 54개를 새로 학습했습니다. ARIMA는 기존에 검증 CRPS로 선택한 **6개 시계의 결과**를 재사용했으며, 선택된 차수는 2종입니다. 확률 LightGBM의 세 시드 예측은 현재 설정에서 동일했고, 신경망은 평균·표본 표준편차를 보고합니다. 시드를 합친 앙상블은 아닙니다.

최신 실행은 `crps_multiseed_20261004T175005+0900`입니다. EMFN의 6개 시계 평균 CRPS는 **1.4642 ± 0.0126**, 허들 분위수 LightGBM은 **1.4607**입니다. EMFN은 +1h/+3h 평균 CRPS가 가장 낮았고, 허들 LightGBM의 영발전 Brier는 6개 시계 모두 더 낮았습니다. 전반적인 확률 예측 우월성은 입증되지 않았습니다. 시간 블록 구간은 개발 기간의 탐색적 분석입니다.

## 현재 논문 구성안

**실험한 모델 목록과 본문에 배치할 모델 목록을 구분합니다.** 본문은 표준 모델 중심으로 간추리고, 이미 수행한 확률 비교는 부록·보조 실험에 보존합니다. CRPS 주 지표와 영발전 Brier 핵심 보조 지표는 유지합니다.

- 현재 조건이 통일된 본문 표: **EMFN, ARIMA, CNN-LSTM, 일반 LightGBM(24h)**. 비교 모델은 3개이며, 최대 5개는 상한이지 채워야 하는 수가 아닙니다.
- 보조 확률 표: **EMFN, 분위수 LightGBM, 허들 분위수 LightGBM**. 본문에서도 해당 비교의 요점과 한계를 요약합니다.
- Persistence는 단순 기준선 추가 후보입니다. 기존 단일 시드 표에는 있으나 현재 다중 시드 표에 임의로 합치지 않았습니다. LSTM·TCN 등 기존 결과도 같은 원칙으로 보존합니다.

[본문용 표](reports/tables/paper_main_standard_baselines.csv), [보조 확률 표](reports/tables/paper_supplementary_probability.csv)는 [표시 설정](configs/paper_presentation.yaml)에 따라 **검증된 기존 결과의 행만 추출**합니다. 모델 재학습·선택·점수 수정은 하지 않습니다. 자세한 기준은 [후속 계획](reports/baseline_benchmarks_and_future_plan.md)에 있습니다. `paper_five_comparators.csv`는 이전 단일 시드 구성안의 파일명으로, 최신 본문 표가 아닙니다.

## 평가 규약과 열 의미

- 주 지표 **CRPS**, 핵심 보조 **영발전 Brier**, 추가 지표 **RMSE·MAE·영발전 AUPRC**를 사용합니다. 확률 모델은 RMSE에 평균, MAE에 중앙값을 사용합니다.
- 결정론 모델은 자체 점출력을 [0,21]로 clip합니다. Dirac CRPS는 MAE와 같고, 확률 출력이 없으면 Brier/AUPRC는 N/A입니다. 이들만으로 확률 예측 우수성을 입증하지 않습니다.
- ARIMA는 가우시안 예측분포를 [0,21]로 censor한 기준선입니다. 분위수 LightGBM은 정렬·범위 제한·끝점 보간을 사용합니다. 이 분포들의 평균은 단순히 raw 평균을 clip한 값과 다를 수 있습니다. [구성 상세](reports/probabilistic_baselines.md)
- 최신 학습 선택은 검증 CRPS입니다. EMFN 학습률 스케줄러도 CRPS를 관찰하지만 학습 손실은 기존 composite loss, CNN-LSTM은 Huber를 유지합니다.
- 시드 SD는 학습 초기화의 변동, 24h/168h paired 블록 구간은 고정된 학습 실행을 조건으로 한 평가 시계열 변동입니다. 다중 비교·계절 비정상성·독립 검증 문제를 해결한 구간은 아닙니다.

| 결과 파일 | `mae`의 의미 | 선택/실행 범위 |
|---|---|---|
| `comprehensive_extended_metric_matrix.csv` | EMFN 평균의 MAE; 중앙값은 `mae_median` | 기존 10개 모델, 신경망 val loss, seed 42 |
| `probabilistic_extended_benchmark.csv` | 확률 모델 중앙값; 평균은 `mae_mean` | 기존 신경망 재사용 + 신규 모델 val CRPS |
| `paper_five_comparators.csv` | 위 확장 표의 중앙값 MAE를 발췌; `mae_mean` 열 없음 | 이전 본문 구성안 |
| `multiseed_crps_metrics.csv` | 확률 모델 중앙값; 평균은 `mae_mean` | 통일된 val CRPS, 검증/평가 및 시드별 기록 |
| `multiseed_crps_summary.csv`, 새 본문/보조 표 | `mae_mean`은 **중앙값 MAE의 시드 평균**; `mae_sd`는 시드 SD | 요약 열의 `_mean`은 분포 평균을 뜻하지 않음 |

## 구조와 데이터

공유 TCN에서 magnitude/zero adapter로 나뉘고 HF/AR 경로와 선택적 기상 특징 MLP를 사용합니다. `sigmoid(z_zero)`는 변수명과 달리 **p=P(Y>0|X)**입니다. 양수 조건부 분포는 `Y/21 ~ Beta(alpha,beta)`이고, 전체 평균은 `21*p*alpha/(alpha+beta)`입니다. 0에 질량이 있지만 21에 별도 점질량은 없습니다.

유계 구간과 영발전 질량을 반영한 통계 모델이며 제조사의 시동·정지 제어 법칙이나 기상 변수의 인과 효과를 식별하지 않습니다. GroupNorm은 입력 윈도우 안의 시간 위치를 함께 정규화하므로 모든 중간 토큰의 엄격한 prefix causality도 주장하지 않습니다.

학습은 2023-01-01 01:00~2024-06-30 23:00, 검증은 2024년 하반기, 평가는 2025년입니다. 발전량은 MWh, 상한은 21 MWh, 영발전은 정확한 관측 0입니다. 기상 표준화는 학습 구간에만 적합합니다. 미래 참조 없이 최대 3시간 ffill하고, 남은 결측 윈도우는 시간축을 압축하지 않고 제외합니다. 관측이 시간 구간 종료 즉시 가용하다는 오프라인 가정의 실제 타당성은 미확인입니다.

시퀀스 모델과 최신 LightGBM은 발전량+기상 6종의 24시간 입력입니다. **이전 LightGBM(+Weather)은 최대 168h lag·달력 특징을 사용한 별도 기준선**입니다. ARIMA는 발전량 과거 상태를 누적합니다. 공통 평가 표본 수는 시계 순서 [1,3,6,9,12,24]h에서 [8736,8734,8731,8728,8725,8713]이며, 모델·시드 간 target/issue 시각을 일치시켰습니다.

전체 파일은 28,440행 중 유효 발전량 28,368개이며 2026년 일부도 포함합니다. 2023~2025는 26,303개, 정확한 0은 4,799개(18.25%)입니다. 최소 양수는 0.024316 MWh입니다. [데이터 QA](reports/tables/data_quality_by_scope.csv)

공식 단위·집계 시각·총/순발전량 구분, 배포 지연, 정격·정비·출력제어 이력은 원자료 명세 확인이 필요합니다. 2026 AWS 미확보를 ffill로 대신하지 않습니다. 풍속 조건부 오차나 임의 비대칭 오차를 실제 cut-in 검증·시장 수익으로 해석하지 않습니다.

## 재현과 파일 안내

Windows 호환 환경은 [설치 안내](reports/probabilistic_baselines.md)의 프로젝트 로컬 런타임을 사용합니다. 저장소 루트에서:

```powershell
$env:PYTHONPATH = (Join-Path $PWD '.experiment_archive/python_runtime') + ';' + $PWD
python -m unittest discover -s test -p "test_*.py" -v
# 기존 검증 결과로 현재 본문/부록 표만 추출 (학습 없음)
python -m src.experiments.export_paper_views
# 새 다중 시드 실험이 필요한 경우에만 실행
python -m src.experiments.run_multiseed_crps
python -m src.experiments.analyze_multiseed_crps
python -m src.experiments.report_multiseed_crps
python -m src.experiments.export_paper_views
```

과거 재현용 `run_comprehensive_extended_matrix`와 `run_comprehensive_ablation_matrix`는 명시적으로 loss 선택 규약을 유지합니다. 전자는 전처리 데이터와 이전 결과 표도 갱신하므로 단순 보고서 열람용 명령이 아닙니다. 과거와 현재 실험을 섞어 선택 효과만의 어블레이션으로 해석하지 않습니다.

- 최신 모델/예측/epoch 이력: [실행 manifest](reports/multiseed_crps_manifest.json)가 가리키는 `models/runs/<run_id>/`. 기존 `models/checkpoints/emfn_*.pt`는 이전 단일 시드 가중치입니다.
- 실행 당시 코드·설정은 run의 `source.zip`과 해시로 보존합니다. 현재 파일은 문서/코드 정합성 수정으로 달라질 수 있습니다. 중단 실행 재개는 당시 코드·설정이 동일할 때만 가능합니다.
- [실험 이력](reports/EXPERIMENT_HISTORY.md), [남은 과제](reports/baseline_benchmarks_and_future_plan.md), [v1/v2/v3 구조](src/models/history/README.md).
- `notebooks/02_baseline_benchmarks_and_failure_modes.ipynb`는 다중 시드와 현재 표시용 표를 읽습니다. 다른 노트북의 기존 그림·수치는 이전 단계 자료입니다.
- 로컬 아카이브·run 본체는 Git에서 제외하며 [수정 전 해시 인덱스](reports/archive/pre_correction_manifest.json)는 추적 대상으로 유지합니다. 문서 위치 이동은 결과 삭제나 평가 독립성 확보를 뜻하지 않습니다.
