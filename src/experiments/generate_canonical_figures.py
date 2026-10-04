"""Regenerate figures from recorded corrected predictions; no training or sample selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data.forecast_protocol import ROOT
from src.utils.metrics import hurdle_beta_quantiles


def main():
    manifest=json.loads((ROOT/'reports/corrected_run_manifest.json').read_text(encoding='utf-8'))
    run=ROOT/'models/runs'/manifest['run_id']
    figures=ROOT/'reports/figures'; figures.mkdir(exist_ok=True)
    # The one-time archive migration is complete. Regeneration must not move
    # unrelated newer figures when the ignored archive is absent in a fresh clone.
    generated=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110})
    def save(fig,name):
        fig.savefig(figures/name,dpi=180,bbox_inches='tight');plt.close(fig)
        generated.append(name)
    bench=pd.read_csv(run/'benchmark.csv')
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for name,d in bench.groupby('model',sort=False):
        d=d.sort_values('horizon_hours')
        for ax,col in zip(axes,['mae','rmse']):
            ax.plot(d.horizon_hours,d[col],marker='o',linewidth=2.5 if name=='EMFN (Proposed)' else 1.2,label=name)
            ax.set(xlabel='Forecast horizon (hours)',ylabel=col.upper()+' (MWh)',xticks=[1,3,6,9,12,24]);ax.grid(alpha=.2)
    axes[1].legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left')
    fig.suptitle('2025 development benchmark | common timestamps within each horizon | seed 42')
    save(fig,'multi_horizon_performance_decay_curve.png')
    for h in [1,3,12,24]:
        f=pd.read_parquet(run/f'emfn__proposed_{h}h_common.parquet')
        # Fixed calendar window chosen for all horizons; not selected by forecast quality.
        d=f[f.target_time.between('2025-01-10','2025-01-17',inclusive='left')]
        lower=hurdle_beta_quantiles(d.p_pos.to_numpy(),d.alpha.to_numpy(),d.beta.to_numpy(),.05)
        upper=hurdle_beta_quantiles(d.p_pos.to_numpy(),d.alpha.to_numpy(),d.beta.to_numpy(),.95)
        fig,axs=plt.subplots(2,1,figsize=(13,6),sharex=True,gridspec_kw={'height_ratios':[3,1]},layout='constrained')
        axs[0].fill_between(d.target_time,lower,upper,alpha=.2,color='#2878a8',label='90% predictive interval')
        axs[0].plot(d.target_time,d.y_true,color='#212121',label='Observed',lw=1)
        axs[0].plot(d.target_time,d.y_pred_clipped,color='#d06a24',label='Mixture mean',lw=1)
        axs[0].set(ylabel='Hourly generation (MWh)',ylim=(0,21),title=f'EMFN +{h}h | 2025 development benchmark | fixed Jan 10-16 window')
        axs[0].legend(ncol=3,fontsize=9)
        axs[1].plot(d.target_time,1-d.p_pos,label='P(Y=0)',color='#2878a8')
        axs[1].scatter(d.target_time,d.y_true.eq(0),s=5,color='black',alpha=.4,label='Observed zero label')
        axs[1].set(ylabel='Zero probability',ylim=(-.05,1.05));axs[1].legend(ncol=2,fontsize=8)
        save(fig,f'emfn_test_forecast_+{h}h.png')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    d=bench[bench.horizon_hours==1].sort_values('bound_violation_pct')
    ax.barh(d.model,d.bound_violation_pct,color='#2878a8')
    ax.set(xlabel='Raw output outside [0, 21] (%)',title='+1h raw physical violations | point errors use clipped forecasts')
    save(fig,'raw_boundary_violations.png')
    f=pd.read_parquet(ROOT/'data/processed/merged_dataset.parquet')
    d=f[(f.datetime.dt.year<=2025)&~f.aws_wind_speed_is_missing&f.generation_mwh.notna()]
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    hb=axs[0].hexbin(d.aws_wind_speed,d.generation_mwh,gridsize=55,mincnt=1,bins='log',cmap='viridis')
    fig.colorbar(hb,ax=axs[0],label='Observed hours (log colour scale)')
    axs[0].set(xlabel='AWS observed wind speed (m/s)',ylabel='Generation (MWh)',title='2023-2025 observed pairs (imputed wind excluded)')
    bins=pd.read_csv(ROOT/'reports/tables/zero_wind_bin_analysis.csv')
    axs[1].bar(bins.wind_bin,100*bins.zero_probability,color='#2878a8')
    axs[1].set(xlabel='AWS wind speed bin (m/s)',ylabel='Zero generation (%)',title='Association only; turbine state is unobserved')
    save(fig,'generation_eda_overview.png')
    abpath=ROOT/'reports/tables/emfn_comprehensive_ablation_matrix.csv'
    if (ROOT/'reports/corrected_ablation_manifest.json').exists():
        a=pd.read_csv(abpath)
        names=['EMFN (Proposed)','EMFN (Endogenous Only)','EMFN (w/o HF Skips)','EMFN (w/o Selective Gate)','EMFN (Deterministic Regression)']
        labels=['Full EMFN','No weather','No HF/AR','No gate','Regression*']
        fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained')
        for i,h in enumerate([1,3]):
            d=a[a.horizon_hours==h].set_index('model').loc[names]
            for ax,col,title in zip(axs,['mae','zero_auprc','bound_violation_pct'],['Clipped MAE (MWh)','Zero AUPRC (N/A for regression)','Raw bound violations (%)']):
                ax.bar(np.arange(5)+(i-.5)*.35,d[col],width=.35,label=f'+{h}h')
                ax.set_xticks(np.arange(5),labels,rotation=25,ha='right');ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2)
        fig.suptitle('Corrected architecture ablations | *Regression also removes zero route/gate and changes loss')
        save(fig,'emfn_ablation_comparison.png')
    (figures/'figure_manifest.json').write_text(json.dumps({'run_id':manifest['run_id'],'fixed_sample_window':['2025-01-10','2025-01-17 exclusive'],'figures':sorted(generated)},indent=2),encoding='utf-8')
    print('Figures regenerated from',run)

if __name__=='__main__': main()
