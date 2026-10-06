"""Generate the five thin, independently runnable research notebooks.

Run this before execute_notebooks. Model/data/evaluation logic is imported,
never duplicated into notebook cells. Old notebooks are archived once locally.
"""
from datetime import datetime
import json
import shutil
import nbformat as nbf
from src.data.forecast_protocol import ROOT
from src.experiments.run_comprehensive_extended_matrix import sha256

BOOT = '''from pathlib import Path
import os, sys

# Google Colab 자동 감지 및 환경 설정
if 'google.colab' in sys.modules or (os.path.exists('/content') and not (Path.cwd() / "configs/base_config.yaml").exists()):
    repo_dir = Path('/content/Wind-Power-Forecasting')
    if not repo_dir.exists():
        print("Google Colab 환경 감지: Wind-Power-Forecasting 저장소를 클론합니다...")
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/lab-axis/Wind-Power-Forecasting.git", str(repo_dir)], check=True)
        print("필수 의존 패키지를 설치합니다 (colab/requirements-colab.txt)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo_dir / "colab/requirements-colab.txt")], check=True)
    os.chdir(repo_dir)
    sys.path.insert(0, str(repo_dir))
    try:
        get_ipython().run_line_magic('cd', str(repo_dir))
    except Exception:
        pass

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs/base_config.yaml").exists())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
runtime = ROOT / ".experiment_archive/python_runtime"
if runtime.exists(): sys.path.insert(0, str(runtime))
from IPython.display import display
from src.workflows import research as wf
%matplotlib inline'''


def main():
    notebooks=ROOT/'notebooks';archive=ROOT/'.experiment_archive/notebooks_before_sync_20261005'
    old=list(notebooks.glob('*.ipynb'))
    if not archive.exists():
        archive.mkdir(parents=True)
        hashes={}
        for p in old:shutil.copy2(p,archive/p.name);hashes[p.name]=sha256(p)
        (archive/'manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    specs={
    '01_sources_and_preprocessing.ipynb':[
        ('m','# 01 · 원자료 수집과 전처리 재현\n\n상명풍력 발전량 + 새별오름 AWS 883 → 시간별 병합 데이터. 이 문서는 이미 수집한 원자료의 위치·무결성·변환을 재현합니다. 자동 다운로드나 수집되지 않은 메타데이터를 만들어내지 않습니다. 모든 계산은 `src/data/*.py`, 호출 안내는 `src/workflows/research.py`에 있습니다.\n\n**실행:** `Python (wind_power)` 커널에서 위→아래 Run All. 노트북은 각각 독립 실행할 수 있습니다. 실제 재학습은 03/04의 명시적 옵션에서만 시작합니다.'),
        ('c',BOOT),('c','wf.environment()'),
        ('m','## 1. 데이터 출처와 로컬 원본\n\n- 발전량: [한국중부발전 상명풍력 발전 실적](https://www.data.go.kr/data/15119454/fileData.do). `data/raw/generation/`의 wide CSV(기준일, 1시~24시).\n- 기상: [기상청 AWS](https://data.kma.go.kr/), 새별오름 883. `data/raw/weather/saebyeol_883-hour/`의 2023·2024·2025 시간 CSV.\n- 분 자료의 시간자료 재구성 감사는 `src/experiments/audit_aws_reconstruction.py` 및 `reports/aws_reconstruction_validation.json`에 별도로 보존합니다. 2026 후보 기상은 본 실험에 병합하지 않습니다.\n\n아래 SHA-256은 **현재 파일**의 식별자입니다. 다운로드 당시 제공기관 설명·정비 이력·전달 지연을 증명하는 값은 아닙니다.'),
        ('c','wf.source_inventory()'),
        ('m','## 2. 변환 규칙과 확인되지 않은 가정\n\n발전량 단위와 시간 집계 경계는 공식적으로 확정되지 않았습니다. 원자료의 스케일 불연속을 바탕으로 코드가 사용하는 가정을 아래에 명시합니다. 21 MWh 상한은 21 MW × 가정한 1시간 구간입니다. 기관 설명이 확인되면 가정과 데이터 버전을 재검토해야 합니다.'),
        ('c','wf.protocol_table()'),
        ('m','## 3. 원본 → generation_hourly → weather_hourly → merged_dataset\n\n`load_generation.build_and_save_generation_interim()` → `process_saebyeol_weather.build_and_save_master_dataset()`를 호출합니다. 24시는 다음 날 00시로 변환하고, 영발전 0은 그대로 보존합니다. 기상은 정규 시간격자에서 과거값만 최대 3시간 채우며 원래 결측·보간 여부·관측 나이를 기록합니다.\n\n기존 데이터는 덮어쓰지 않습니다. `.experiment_archive/notebook_reproduction/`에 다시 만들고 기존 45개 열 전체와 정확히 일치하는지 검사합니다. 이 단계는 파일 전체를 변환하지만 2026년 예측·성능 평가는 하지 않습니다.'),
        ('c','wf.rebuild_preprocessing()'),
        ('m','## 4. 연구 구간의 결측과 분할\n\n학습: 2023~2024 상반기, 검증: 2024 하반기, 개발 평가: 2025. 아래 요약과 다음 노트북 EDA는 2025년 말까지만 사용합니다. 2025년은 여러 차례 모델 개발에 참조한 구간이므로 독립 테스트라고 부르지 않습니다.'),
        ('c','wf.split_quality()'),('c','wf.weather_quality()'),
        ('m','**다음:** [02 · 실제 모델 입력 EDA](02_model_dataset_eda.ipynb). 상세 구현: `src/data/load_generation.py`, `src/data/process_saebyeol_weather.py`, `src/data/forecast_protocol.py`.')],
    '02_model_dataset_eda.ipynb':[
        ('m','# 02 · 실제 모델 입력 데이터셋 EDA\n\n전처리된 파일 전체와 모델이 실제로 사용할 수 있는 윈도우는 다릅니다. 이 노트북은 학습·검증·개발 평가 분할과 24시간 입력의 유효 표본을 확인합니다. 기상과 발전량의 관계는 관측상 연관이며 인과 효과나 개별 터빈 파워 커브가 아닙니다.'),
        ('c',BOOT),('c','wf.split_quality()'),
        ('m','## 1. 시계열·분포·영발전과 관측소 풍속\n\n바람 그림은 보간하지 않은 관측 기상만 사용합니다. 그림을 보고 2025년을 다시 튜닝하는 절차는 포함하지 않습니다.'),
        ('c','wf.plot_dataset();'),
        ('m','## 2. 공통 평가 표본\n\n발행 시각 t의 입력은 [t−23h, t], 타깃은 t+h입니다. 배열의 exclusive index i는 입력 [i−24:i], 타깃 i+h−1에 대응합니다. 결측이 있으면 행을 삭제해 시간을 당기지 않고 해당 윈도우를 제외합니다. 분할 경계마다 윈도우를 새로 시작하므로 검증·평가의 첫 24시간을 과거 구간에서 보충하지 않습니다. 모든 모델을 같은 시각에 비교하기 위해 비기상 모델에도 이 공통 표본 규칙을 적용합니다.'),
        ('c','wf.window_counts()'),
        ('m','## 3. 실제 한 개 입력과 타깃\n\n아래 발전량 입력은 용량으로 나누고 기상 6채널은 학습 구간의 평균·표준편차로 변환했습니다. EMFN Dataset은 발전량을 MWh로 반환한 뒤 `EMFN.forward()`에서 /21을 수행하고, 일반 신경망 Dataset은 /21을 미리 수행합니다. 같은 물리 단위를 두 번 나누지 않습니다. 24h LightGBM은 기존 규약대로 MWh 발전량을 사용합니다.'),
        ('c','X, target = wf.input_example(split="validation", horizon=6)\ndisplay(X)\ndisplay(target)'),
        ('m','## 4. 코드 추적 경로\n\n`prepare_splits()` → train-only weather scaler → `valid_window_indices()` → `WindTimeSeriesDataset` / `EMFNMultiChannelDataset` / `flattened_windows()`. 입력 특징은 명시적인 7채널이며 `is_zero`, 원본 단위, 타깃 시각 풍속 등 정답·감사 열은 입력에 들어가지 않습니다.\n\n**다음:** [03 · 베이스라인 학습과 평가](03_baseline_training_and_metrics.ipynb).')],
    '03_baseline_training_and_metrics.ipynb':[
        ('m','# 03 · 베이스라인 학습과 성능 지표\n\n현재 구현된 14개 베이스라인과 EMFN, 총 15개 모델 계열을 동일 평가 시점에 재실험합니다. 6개 시계(1/3/6/9/12/24h), 학습 시드 42/2026/3407을 사용합니다. Persistence와 ARIMA는 시드 반복을 만들지 않습니다. 모델별 입력 예산과 출력 형태는 아래 표로 구분합니다. 과거 v1/v2와 구조 어블레이션은 이 베이스라인 목록에 포함하지 않습니다.'),
        ('c',BOOT),('c','wf.model_catalog()'),
        ('m','## 1. 고정된 학습 규약\n\n설정: `configs/all_baselines.yaml`, 실행: `src/experiments/run_all_baselines.py`, 생성자: `src/models/benchmark_registry.py`. 신경망 체크포인트와 트리 반복 수, ARIMA 차수는 검증 CRPS로 선택합니다. 결정론 CRPS는 clipped MAE입니다. 학습 손실은 EMFN NLL+Huber, 다른 신경망 Huber, 점예측 LightGBM L2를 유지합니다.\n\nGRU의 마지막 ReLU는 제거해 음수 초기 출력에서도 학습 기울기가 흐르도록 수정했습니다. 새 결과는 수정 전 0 수렴 실행과 구분합니다. 장기 lag LightGBM은 NaN을 자체 처리하고 최대 168h lag·달력 특징을 사용하므로 24h 입력 모델과 구조만의 비교가 아닙니다. Prophet은 기존 연간/주간/일간 계절성 설정의 train-only 달력 예측이며 rolling refit을 하지 않습니다. 학습 자료가 2년 미만이므로 연간 계절성 식별의 한계도 있습니다.'),
        ('c','wf.run_status()'),
        ('m','## 2. 재학습 실행 (선택)\n\n기본값 False는 검증된 저장 결과를 읽는 모드입니다. True이면 **EMFN을 포함한 전체 행렬을 처음부터** 새 run 폴더에 학습하며 상당한 시간이 걸립니다. 234는 모델×시계×시드 조합 수이지 독립 적합 함수 호출 수가 아닙니다. ARIMA 후보 5개와 시드별 Prophet은 한 번 적합한 파라미터를 여러 시계에서 사용합니다. 완료 후 검증기를 실행해야 최신 manifest와 보고 표로 승격됩니다.'),
        ('c','RUN_FULL_RETRAINING = False\nif RUN_FULL_RETRAINING:\n    new_run = wf.train_all()\n    wf.publish_run(new_run)'),
        ('m','## 3. 검증에 의한 선택과 학습 실패 진단\n\n아래 표의 검증 예측 표준편차가 0이면 상수 예측 경고입니다. 경고를 숨기거나 2025 성능을 보고 좋은 시드만 고르지 않습니다. 선택 없는 Persistence/Prophet은 그 사실을 그대로 기록합니다.'),
        ('c','wf.validation_diagnostics()'),
        ('m','## 4. 베이스라인 결과\n\nCRPS ↓, 영발전 Brier ↓, RMSE ↓, MAE ↓, 영발전 AUPRC ↑ 순서입니다. 확률 모델의 RMSE는 예측 평균, MAE는 예측 중앙값입니다. 점예측 모델의 CRPS는 MAE이고 확률 지표는 N/A입니다. 표시한 ±는 시드별 지표의 표본 표준편차이며 앙상블 점수가 아닙니다. 단일 결정론 실행의 SD는 계산하지 않습니다.'),
        ('c','baselines = [name for name in wf.ALL_MODELS if name != "EMFN (Proposed)"]\nwf.display_scores(models=baselines, horizons=(1, 6, 24))'),
        ('m','모든 시계·시드·구현 지표를 읽으려면 아래 DataFrame을 사용합니다. 파일: `reports/tables/all_baselines_metrics.csv`(468행), `all_baselines_summary.csv`(180행). Notebook의 접힌 표시가 결과 누락을 뜻하지 않습니다.'),
        ('c','wf.metric_definitions()'),
        ('c','all_baseline_metrics = wf.metrics_table(models=baselines, horizons=None, full=True)\ndisplay(all_baseline_metrics)\nprint("Available metric columns:", all_baseline_metrics.columns.tolist())'),
        ('m','**다음:** [04 · EMFN 구조와 학습](04_emfn_architecture_and_results.ipynb).')],
    '04_emfn_architecture_and_results.ipynb':[
        ('m','# 04 · 제안 모델 EMFN v3의 구조·학습·결과\n\n주 구현: `src/models/emfn.py`. 손실·학습: `hurdle_beta.py`, `emfn_trainer.py`. 아래 구조와 파라미터 표는 실제 생성자를 호출합니다. 이 모델은 유계 출력과 영발전 질량을 갖는 통계적 확률 모델이며 제조사 cut-in/cut-out 제어 법칙이나 인과 효과를 직접 식별하지 않습니다.'),
        ('c',BOOT),('c','wf.plot_architecture();'),('c','wf.architecture_table()'),
        ('m','## 1. 허들 출력과 학습 목적\n\n입력 발전량은 항상 C=21로 나눕니다. 공유 TCN → magnitude/zero adapter 두 분기. magnitude는 HF·AR, zero는 기상 및 과거 영발전 요약 특징을 추가합니다. GroupNorm이 윈도우 안 시간축을 함께 사용하므로 중간 토큰의 엄격한 prefix causality를 주장하지 않습니다. 발행 시점 이후 관측은 입력에 넣지 않습니다.\n\n- 코드의 `z_zero` 이름과 달리 $p=\\sigma(z_{zero})=P(Y>0\\mid X)$.\n- $P(Y=0)=1-p$, $Y/C\\mid Y>0\\sim Beta(\\alpha,\\beta)$.\n- $E[Y\\mid X]=C p\\alpha/(\\alpha+\\beta)$; MAE에는 혼합분포 중앙값을 사용합니다.\n- 양수에서 Beta NLL + Bernoulli NLL + 2×정규화 평균의 Huber loss. eps는 수치 안정화에 쓰이며 영발전 라벨은 정확한 0입니다.\n- 0에는 점질량이 있고 C에는 별도 점질량이 없습니다. 이는 예측 불확실성이며 베이지안 모수 사후분포가 아닙니다.'),
        ('m','## 2. 단일 EMFN 재학습 (선택)\n\nFalse가 기본입니다. True로 바꾸면 고정된 설정의 +1h/seed42를 직접 학습하고 별도 로컬 실험 폴더에 저장합니다. 본문 점수는 03에서 생성·검증한 전체 실행만 사용하며 여기의 추가 실행으로 좋은 결과를 골라 바꾸지 않습니다.'),
        ('c','RUN_EMFN_EXAMPLE = False\nif RUN_EMFN_EXAMPLE:\n    display(wf.train_emfn_example(horizon=1, seed=42))'),
        ('m','## 3. 검증 CRPS 체크포인트 선택\n\n예시는 사전에 정한 +6h/seed42입니다. epoch별 검증 CRPS와 선택 epoch는 해당 checkpoint의 `.pt.history.json`에서 읽습니다.'),
        ('c','wf.plot_training(model="EMFN (Proposed)", horizon=6, seed=42);'),
        ('c','wf.display_scores(models=["EMFN (Proposed)"], horizons=None)'),
        ('m','## 4. 예측 평균·중앙값·구간\n\n아래는 +6h/seed42의 개발 평가 첫 7개 가용 일자입니다. 잘 맞는 구간을 사후 선정한 그림이 아닙니다. 구간이 좁은 것만으로 보정이 좋다고 해석하지 않고 포함률과 함께 검토합니다.'),
        ('c','wf.plot_forecast(horizon=6, seed=42);'),
        ('m','## 5. 구조 기여 검증의 현재 경계\n\n`src/models/emfn_ablation.py`의 shared_adapter / 기상 상태 특징 제거 / HF·AR 제거 / 공통 표현 구조의 linear·sigmoid·hurdle 대조군은 구현·합성 검증 상태입니다. 실제 126개 구조 학습은 아직 수행하지 않았습니다. 과거 `regression_mode`는 Route B와 gate도 제거하므로 순수 출력 헤드 비교가 아닙니다. 이 노트북은 준비 상태를 완료 결과처럼 보여주지 않습니다.\n\n**다음:** [05 · 전체 비교와 논문용 정리](05_comparison_and_research_limits.ipynb).')],
    '05_comparison_and_research_limits.ipynb':[
        ('m','# 05 · 전체 모델 비교와 논문용 결과 정리\n\n모든 표·그림은 동일한 새 실행의 검증된 저장 예측을 사용합니다. 본문 배치는 표준 모델 중심으로 간추리되 전체 15개 모델 결과와 확률 비교를 남깁니다. 표시 모델을 간추리는 것은 편집 결정이며 불리한 결과를 제거하거나 실험 전에 정한 것처럼 소급하는 절차가 아닙니다.'),
        ('c',BOOT),('c','wf.run_status()'),
        ('m','## 1. 공통 시점 확인\n\n모든 모델·시드의 target_time, issue_time, y_true를 대조합니다. ARIMA·Persistence에도 동일한 공통 표본을 적용합니다. 데이터 누락이나 horizon 차이를 행 순서만으로 맞추지 않습니다.'),
        ('c','wf.verify_common_samples()'),
        ('m','## 2. 본문 표: 5개 모델, 대표 3개 시계\n\n**EMFN + Naive Persistence + ARIMA + LSTM + 일반 LightGBM(24h)**. LightGBM 계열은 본문에서 한 모델만 사용합니다. CRPS/Brier/RMSE/MAE/AUPRC를 나열하고 +1h/+6h/+24h를 표로, 전체 시계를 곡선으로 제시합니다. N/A는 점출력 모델에 예측 확률이 없다는 뜻입니다.'),
        ('c','wf.display_scores(models=wf.MAIN, horizons=(1,6,24))'),
        ('c','wf.plot_horizons(models=wf.MAIN);'),
        ('m','## 3. 확률 베이스라인과 보정 한계\n\nARIMA, 분위수 LightGBM, 허들 분위수 LightGBM을 비교합니다. 기존 분위수·허들 모델의 수치가 유리하거나 불리해도 보존합니다. EMFN 대 점예측 모델의 CRPS 비교만으로 확률 모델 일반에 대한 우월성을 주장하지 않습니다. 명목 구간 포함률·구간 폭과 Brier를 같이 해석합니다. 점질량이 있는 중앙 예측구간은 보정된 분포에서도 포함률이 정확히 90%가 아닐 수 있습니다.'),
        ('c','probabilistic = ["EMFN (Proposed)", "ARIMA", "Quantile LightGBM (24h matched)", "Hurdle Quantile LightGBM (24h matched)"]\nwf.display_scores(models=probabilistic, horizons=None)'),
        ('c','wf.plot_probability_comparison();'),
        ('m','## 4. 검증 구간 결과와 전체 지표\n\n검증 결과는 모델 선택에 사용한 구간의 점수이지 독립 성능이 아닙니다. 아래 변수에는 본문에서 생략한 모델·시계·전체 계산 지표가 모두 남습니다. raw 물리 위반율은 clipping 전 출력, 점오차는 bounded 출력에 대한 값입니다. 0에 가까운 발전량 때문에 MAPE가 불안정할 수 있으므로 보조 진단으로만 봅니다.'),
        ('c','wf.display_scores(split="validation", models=wf.MAIN, horizons=(1,6,24))'),
        ('c','full_results = wf.metrics_table(full=True, horizons=None)\ndisplay(full_results)\nprint("Full metric rows:", len(full_results))'),
        ('m','## 5. 논문에서 남겨야 할 한계와 다음 단계\n\n1. 2025년은 반복 열람한 개발 벤치마크. 2026년은 기술통계 열람 이력이 있으나 성능 선택에는 미사용이며 아직 평가하지 않았습니다.\n2. 단위·1시간 집계 경계·계량 경계·관측 전달 지연의 기관 확인이 남습니다. 관측소 풍속은 터빈 허브 높이 풍속과 같다고 가정하지 않습니다.\n3. 서로 다른 입력 예산(24h/168h/history/calendar), 학습 손실·예산을 명시합니다. Prophet의 train-only 달력 예측은 최근 상태를 반영하는 rolling-origin 모델과 정보 사용이 다릅니다.\n4. 시드 SD는 학습 변동입니다. 기존 실행의 paired block 구간을 새 실행의 검정 결과로 재사용하지 않습니다.\n5. 다중 시드 구조 어블레이션, 기상 변수군 기여, 최종 설계 고정 후 후속 기간 평가가 남습니다. 예측 정보 기여를 인과 효과로 바꾸어 주장하지 않습니다.\n\n재현 추적: `reports/all_baselines_manifest.json` → `models/runs/<run_id>/source.zip`, 가중치, epoch 이력, validation/test 예측 → `reports/all_baselines_validation.json` → 최신 CSV. 모든 상세 로직은 `.py`에서 관리하며 노트북에는 중복 학습 구현을 두지 않습니다.')]
    }
    for name,cells in specs.items():
        nb=nbf.v4.new_notebook()
        nb.metadata.kernelspec=dict(display_name='Python (wind_power)',language='python',name='wind_power')
        nb.metadata.language_info=dict(name='python',version='3.11')
        badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-axis/Wind-Power-Forecasting/blob/main/notebooks/{name})\n\n"
        built_cells = []
        for i, (kind, text) in enumerate(cells):
            if i == 0 and kind == 'm':
                lines = text.split('\n', 1)
                augmented = f"{lines[0]}\n\n{badge}{lines[1].lstrip()}" if len(lines) > 1 else f"{text}\n\n{badge}"
                built_cells.append(nbf.v4.new_markdown_cell(augmented))
            elif kind == 'm':
                built_cells.append(nbf.v4.new_markdown_cell(text))
            else:
                built_cells.append(nbf.v4.new_code_cell(text))
        nb.cells = built_cells
        nbf.write(nb, notebooks / name)
    # Remove only old, already archived notebook files inside the verified folder.
    for p in old:
        if p.name not in specs:
            assert p.resolve().parent==notebooks.resolve() and (archive/p.name).exists()
            p.unlink()
    print('Wrote',len(specs),'notebooks; old versions in',archive)


if __name__=='__main__':main()
