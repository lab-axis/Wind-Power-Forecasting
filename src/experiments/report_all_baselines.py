"""Build tables/figures/documentation from the independently verified new run."""
import json
import shutil
import pandas as pd
import matplotlib
if __name__=='__main__':matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data.forecast_protocol import ROOT
from src.workflows import research as wf
from src.experiments.run_comprehensive_extended_matrix import sha256


def markdown_table(df):
    def clean(value):return str(value).replace('|','/')
    lines=['| '+' | '.join(map(clean,df.columns))+' |','|'+'|'.join(['---']*len(df.columns))+'|']
    lines+=['| '+' | '.join(map(clean,row))+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    m,run=wf.load_run();figdir=ROOT/'reports/figures';figdir.mkdir(exist_ok=True)
    figures=[]
    for name,figure in [('all_baselines_main_horizons',wf.plot_horizons()),('all_baselines_probability',wf.plot_probability_comparison()),
                        ('all_baselines_architecture',wf.plot_architecture()),('all_baselines_forecast_example',wf.plot_forecast())]:
        path=figdir/(name+'.png');figure.savefig(path,dpi=140);plt.close(figure);figures.append(path)
    scores=wf.metrics_table(full=True);summary=wf.summary_table(horizons=None)
    per_seed=scores.groupby(['model','seed'],dropna=False).crps.mean().reset_index()
    average=per_seed.groupby('model',sort=False).crps.agg(['mean','std']).reset_index().sort_values('mean')
    average.columns=['model','mean_CRPS_over_six_horizons','seed_SD_of_six_horizon_mean']
    average.to_csv(ROOT/'reports/tables/all_baselines_horizon_average.csv',index=False)
    warnings=[e for e in m['experiments'] if e['validation_constant_warning']]
    chosen=wf.display_scores(models=wf.MAIN,horizons=(1,6,24))
    all_scores=wf.display_scores(horizons=None)
    winner=summary.loc[summary.groupby('horizon_hours').crps_mean.idxmin(),['horizon_hours','model','crps_mean']].sort_values('horizon_hours')
    prob=summary[summary.model.isin(['EMFN (Proposed)','ARIMA',wf.TREES[1],wf.TREES[2]])]
    zero_winner=prob.loc[prob.groupby('horizon_hours').zero_brier_mean.idxmin(),['horizon_hours','model','zero_brier_mean']].sort_values('horizon_hours')
    report=f'''# 전체 베이스라인 재실험과 노트북 동기화

실행 `{m['run_id']}`. **15개 모델 계열, 6개 시계, 234개 모델·시계·시드 조합, 검증/개발 평가 468행**을 새로 계산했습니다. 완료된 기존 run의 가중치·예측을 재사용하지 않았으며 2026년 예측 성능을 계산하지 않았습니다. 연구의 주 지표는 CRPS, 핵심 보조 지표는 영발전 Brier입니다.

## 학습 범위와 변경

- 신경망 7종 × 6시계 × 3시드 = **126개 학습**: EMFN, DLinear, LSTM, GRU, CNN-LSTM, TCN, Transformer.
- LightGBM 4종 × 6시계 × 3시드 = **72개 모델 묶음 학습**: 168h lag·달력 기준선, 24h 점예측, 24h 직접 분위수, 24h 허들 분위수. 분위수 모델 묶음은 여러 개 트리를 적합합니다.
- ARIMA 후보 **5개를 새로 적합**해 수렴 후보 중 각 시계의 검증 CRPS로 차수를 선택했습니다. 파라미터는 고정하고 각 발행 시각까지의 관측으로 상태만 필터링합니다.
- Prophet은 **3개 시드별 적합**, 동일한 train-only 달력 예측을 6시계에서 각각 평가했습니다. 일간·주간·연간 계절성, changepoint prior 0.05를 유지합니다. 최근 관측을 매 시점 다시 적합하는 rolling baseline이 아니며 학습 기간이 2년 미만이어서 연간 계절성 식별에 한계가 있습니다. Prophet의 구간 출력은 사용하지 않은 점예측 평가입니다.
- Naive/Diurnal Persistence **12개 시계 조합**을 공통 표본에서 다시 계산했습니다. 결정론 모델의 시드를 복제해 SD를 만들지 않습니다.
- **GRU의 마지막 ReLU를 선형 출력으로 변경**했습니다. 음수 초기 출력에서 기울기가 끊기는 경로를 제거하며 공통 평가기의 clipping은 유지합니다. 수정 전 GRU의 0 수렴 결과와 별도 실행으로 기록합니다.
- 장기 lag LightGBM은 검증 CRPS early stopping과 NaN 자체 처리를 적용했습니다. 24h 공통 표본은 유지하지만 최대 168h lag와 달력 특징을 쓰므로 입력 예산이 다릅니다. 과거 실행 대비 차이를 단일 수정의 효과로 해석하지 않습니다.

시드: 42·2026·3407. 신경망·트리 선택은 검증 CRPS, Persistence는 고정 규칙, Prophet은 고정 설정입니다. 손실 함수와 학습 예산이 완전히 동일한 아키텍처 실험은 아닙니다. 모든 모델의 발행/타깃 시각과 정답을 대조했습니다. 검증 상수 예측 경고는 **{len(warnings)}개**입니다.

## 본문용 간결한 표

표준 모델 중심의 **EMFN, Naive Persistence, ARIMA, LSTM, 일반 LightGBM(24h)** 구성입니다. 이 배치는 결과를 이미 열람한 뒤 정한 편집 결정입니다. CNN-LSTM·확률 LightGBM·나머지 모델도 삭제하지 않고 전체 표에 남깁니다.

{markdown_table(chosen)}

±는 개별 학습 점수의 표본 SD이며 시드 앙상블이 아닙니다. 시드가 없는 실행은 SD를 표시하지 않습니다. N/A는 확률 출력이 없는 모델에 해당 지표를 부여하지 않았다는 뜻입니다. RMSE에는 분포 평균, MAE에는 분포 중앙값을 사용하며 점모델은 자체 clipped 출력을 사용합니다. Dirac CRPS=MAE이므로 확률 모델만의 우월성을 이 비교만으로 입증할 수 없습니다.

![대표 모델의 전체 시계 비교](figures/all_baselines_main_horizons.png)

## 전체 비교의 관찰

모든 15개 모델을 포함한 시계별 평균 CRPS 최저값:

{markdown_table(winner.round(6))}

확률 출력 모델의 시계별 평균 영발전 Brier 최저값:

{markdown_table(zero_winner.round(6))}

순위는 동일 개발 구간의 기술적 비교이며 통계적 유의성·독립 검증의 결론이 아닙니다. 단순 모델을 본문에 배치하더라도 확률 베이스라인 결과를 함께 언급해야 합니다. 구간 포함률과 구간 폭을 같이 해석하며, 어떤 모델이 모든 시계에서 잘 보정됐다고 일반화하지 않습니다. 기존 실행의 paired block 구간은 이 새 실행의 유의성 구간으로 재사용하지 않았습니다.

![확률 모델 비교](figures/all_baselines_probability.png)

## 전체 모델·시계의 주 지표

{markdown_table(all_scores)}

전체 측정 지표는 [시드별 468행](tables/all_baselines_metrics.csv), [180행 요약](tables/all_baselines_summary.csv)에 보존합니다. 요약 접미사 `_mean`/`_sd`는 시드 집계이며, 요약의 `mae_mean`은 중앙값 MAE의 시드 평균입니다. 개별 지표 파일의 `mae_mean`은 분포 평균 예측의 MAE로 의미가 다릅니다. `vpr_pct`는 표준편차 비율, `mae_cut_in`은 관측소 풍속 조건의 진단이며 실제 시동 제어 검증이 아닙니다. `imbalance_loss_mwh`는 설정된 비대칭 오차로 시장 수익이 아닙니다.

## 노트북 구성과 재현

1. [원자료·전처리](../notebooks/01_sources_and_preprocessing.ipynb): 원본 목록/해시, 단위·시각 가정, 별도 경로 재구축과 정확한 값 비교.
2. [입력 데이터 EDA](../notebooks/02_model_dataset_eda.ipynb): 분할, 결측, 영발전, 풍속 연관, 유효 윈도우 및 실제 입력.
3. [베이스라인](../notebooks/03_baseline_training_and_metrics.ipynb): 모델 목록/입력 예산, 전체 재학습 호출, 검증 선택과 결과.
4. [EMFN](../notebooks/04_emfn_architecture_and_results.ipynb): 구조·파라미터, 손실, 단일 학습 호출, CRPS 이력·분포 예측.
5. [전체 비교](../notebooks/05_comparison_and_research_limits.ipynb): 공통 시각 검증, 본문 축약 표, 전체 시계, 확률 비교와 한계.

노트북은 `src/workflows/research.py`를 호출하고 데이터·모델·학습·지표 구현은 기존 `src` 모듈에 둡니다. 기본 실행은 저장 결과를 열람하며 `RUN_FULL_RETRAINING`/`RUN_EMFN_EXAMPLE`만 명시적으로 True일 때 학습합니다. 이전 4개 노트북은 `.experiment_archive/notebooks_before_sync_20261005/`에 보존했습니다.

저장소 루트의 호환 Python 환경에서:

```powershell
$env:PYTHONPATH = (Join-Path $PWD '.experiment_archive/python_runtime') + ';' + $PWD
python -m src.experiments.run_all_baselines
python -m src.experiments.verify_all_baselines --run models/runs/<new_run_id>
python -m src.experiments.report_all_baselines
python -m src.experiments.export_paper_views
python -m src.workflows.execute_notebooks
python -m unittest discover -s test -p "test_*.py"
```

중단 시 같은 코드·설정으로 `run_all_baselines --resume models/runs/<run_id>`를 사용합니다. 모델·데이터·설정은 [실행 manifest](all_baselines_manifest.json)와 각 run의 `source.zip`/SHA-256으로 추적합니다. [산출물 검증](all_baselines_validation.json)은 모든 468행 지표 재계산, 신경망 전 validation/test 재로딩, 다른 적합 모델의 처음/중간/마지막 타깃 재로딩, 공통 시각/정답 대조를 기록합니다.

## 남은 연구 경계

2025는 반복 열람한 개발 벤치마크입니다. 발전량 단위·집계 구간·관측 전달 지연은 확인되지 않은 가정입니다. 기상 효과는 조건부 예측 정보이며 인과 효과가 아닙니다. 현재 모델은 제조사 cut-in/cut-out 곡선을 내장하지 않습니다. 별도 구조 대조군 126개 실제 학습·고정 h 비교·후속 기간 평가는 아직 완료하지 않았습니다. 이번 결과를 보고 2026 평가를 실행하거나 설정을 조정하지 않았습니다.
'''
    path=ROOT/'reports/all_baselines_results.md';path.write_text(report,encoding='utf-8')
    artifacts=figures+[path,ROOT/'reports/tables/all_baselines_horizon_average.csv']
    provenance=dict(run_id=m['run_id'],status='generated_from_verified_run',
        artifact_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in artifacts},
        sources={p:sha256(ROOT/p) for p in ['src/experiments/report_all_baselines.py','src/workflows/research.py']})
    (ROOT/'reports/all_baselines_report_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('Published',path)


if __name__=='__main__':main()
