# 연구 재현 노트북

`Python (wind_power)` 커널을 선택하고 각 노트북을 위에서 아래로 실행합니다. 각각 독립 실행 가능하며 기본 설정에서는 학습하지 않고 검증된 저장 결과를 읽습니다.

| 순서 | 노트북 | 확인할 내용 |
|---|---|---|
| 01 | [원자료·전처리](01_sources_and_preprocessing.ipynb) | 수집 출처, 파일 해시, 단위/시각 가정, 별도 경로 전처리 재현 |
| 02 | [모델 입력 EDA](02_model_dataset_eda.ipynb) | 시계열/분포, 기상 결측, 분할, 입력 윈도우와 타깃 |
| 03 | [베이스라인](03_baseline_training_and_metrics.ipynb) | 15개 모델 계열의 입력/출력, 전체 학습 호출, 선택과 모든 지표 |
| 04 | [EMFN](04_emfn_architecture_and_results.ipynb) | v3 구조, 손실, 학습 호출, 체크포인트 선택, 분포 예측 |
| 05 | [전체 비교](05_comparison_and_research_limits.ipynb) | 본문 5개 모델/대표 시계, 전체 시계·확률 비교, 한계 |

노트북의 짧은 셀은 `src/workflows/research.py`를 호출합니다. 상세 구현은 `src/data`, `src/models`, `src/utils`, 실행 규약은 `src/experiments/run_all_baselines.py`와 `configs/all_baselines.yaml`에 있습니다. 코드 수정 시 `.py`를 고치고 노트북을 다시 실행하면 출력이 갱신됩니다.

- `RUN_FULL_RETRAINING=True`: EMFN 포함 전체 행렬을 새 run에 재학습합니다. 상당한 시간이 필요합니다.
- `RUN_EMFN_EXAMPLE=True`: EMFN 1시계·1시드를 별도 예시 폴더에서 학습합니다. 본문 결과를 대체하지 않습니다.
- `models/runs/`는 Git 제외 대상입니다. 다른 컴퓨터에서는 해당 run을 전달받거나 재학습·검증해야 결과 호출이 가능합니다. 실행 출력이 저장된 노트북과 공개 CSV는 run 본체 없이도 읽을 수 있습니다.
- Windows의 ARIMA 호환 환경은 `requirements_probabilistic.txt` 및 `reports/probabilistic_baselines.md`를 확인합니다. 커널 내부에서 numpy가 이미 import된 상태로 환경을 바꿨다면 커널을 재시작합니다.
- 이전 노트북 4개는 `.experiment_archive/notebooks_before_sync_20261005/`에 보존합니다.
- 2026년 기상/발전량 후보는 이번 모델 학습·평가에서 사용하지 않습니다. 2025년은 독립 블라인드 테스트가 아닌 개발 벤치마크입니다.

일괄 실행: `python -m src.workflows.execute_notebooks`. 생성 템플릿을 바꿀 때만 `python -m src.workflows.build_notebooks`를 실행합니다. 이 명령은 현재 노트북 셀·출력을 다시 생성하므로 노트북에 직접 작성한 메모는 먼저 보존하십시오.
