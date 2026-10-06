"""A-to-Z research inspection APIs; computational definitions remain in src/.

All descriptive plots here stop at 2025. A saved complete run is required for
result inspection. Fresh training is explicit and creates a separate run.
"""
import json
from pathlib import Path
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from src.data.forecast_protocol import ROOT, WEATHER_COLUMNS, prepare_splits, valid_window_indices
from src.experiments.run_comprehensive_extended_matrix import sha256, seed_everything
from src.models.benchmark_registry import ALL_MODELS, NEURAL, TREES, LONG_TREE, make_neural

PRIMARY = ['crps', 'zero_brier', 'rmse', 'mae', 'zero_auprc']
MAIN = yaml.safe_load((ROOT/'configs/paper_presentation.yaml').read_text(encoding='utf-8'))['main_models']


def environment():
    """Record the actually imported versions, including the local Windows overlay."""
    import scipy, torch
    return pd.DataFrame([
        ('python', platform.python_version()), ('numpy', np.__version__),
        ('scipy', scipy.__version__), ('pandas', pd.__version__), ('torch', torch.__version__),
        ('lightgbm', importlib.metadata.version('lightgbm')), ('prophet', importlib.metadata.version('prophet')),
        ('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'),
    ], columns=['component','version'])


def source_inventory():
    """Only training-source files; minute files are reserved for the separate audit."""
    paths=list((ROOT/'data/raw/generation').glob('*.csv'))
    for directory in ['saebyeol_883-hour','saebyeol_883']:
        paths.extend((ROOT/'data/raw/weather'/directory).glob('*.csv'))
    return pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(set(paths))])


def protocol_table():
    return pd.DataFrame([
        ('target', 'generation_mwh; exact observed 0 is the zero label'),
        ('unit assumption', 'before 2025-02-01: raw/1e6; afterwards: raw/1e3; provider confirmation unavailable'),
        ('time assumption', '1-hour energy, interval-end label; 24:00 = next-day 00:00; zero telemetry delay assumed'),
        ('capacity', '21 MWh per assumed one-hour interval'),
        ('input', '24 hours: generation + wind speed, wind sin/cos, temperature, humidity, local pressure'),
        ('missing values', 'hourly grid, weather past-only ffill <=3h; invalid windows excluded without compressing time'),
        ('scaling', 'generation /21; weather mean/std fitted on training split only'),
        ('selection', '2024 H2 validation CRPS; fixed-rule models have no selection'),
        ('evaluation', '2025 development benchmark; not an independent blind test'),
        ('2026', 'not forecast or scored by these notebooks'),
    ],columns=['item','definition'])


def metric_definitions():
    return pd.DataFrame([
        ('crps', 'MWh; lower', 'Full bounded distribution; 1001-point grid, Dirac uses exact MAE'),
        ('zero_brier / zero_auprc / zero_auroc / zero_ece', 'lower / higher / higher / lower', 'Observed Y=0 event; N/A without fitted probabilities'),
        ('rmse / mse', 'MWh / MWh²; lower', 'Predictive mean for distributions; clipped point otherwise'),
        ('mae', 'MWh; lower', 'Predictive median for distributions; clipped point otherwise'),
        ('mae_mean', 'MWh; lower', 'Per-run CSV: MAE of predictive mean, NOT seed average'),
        ('nmae_pct / nrmse_pct', '% of C; lower', 'Both use predictive MEAN for distributions; nmae is not necessarily 100*mae/C'),
        ('rse / rrse / rae', 'dimensionless; lower', 'Relative point errors against reference variability'),
        ('wape_pct / smape_pct / mape_nonzero_pct', '%; lower', 'Mean point forecast; nonzero MAPE excludes exact zeros'),
        ('r2 / corr / mbe / tic', 'dimensionless except MBE (MWh)', 'Point fit, correlation, signed bias and Theil inequality'),
        ('mae_ramp / rmse_ramp / corr_ramp / vpr_pct', 'MWh / MWh / r / %', 'Consecutive 1h pairs only; VPR is prediction/observation STD ratio'),
        ('bound_violation_pct / integrated_negative_mwh / min_pred_mwh / max_pred_mwh', '% / MWh', 'Raw BEFORE clipping; probability models use their recorded raw mean diagnostic'),
        ('zero_f1 / zero_balanced_acc', 'higher', 'Exact-zero event implied by clipped point predictions; distinct from probability scoring'),
        ('zero_point_f1_mean / zero_point_balanced_acc_mean', 'higher', 'Same point-event diagnostic for the predictive mean of probabilistic models'),
        ('picp_90_pct / pinaw_90_pct / winkler_score_90', '% / % of C / MWh', '90% central interval coverage, normalized width, interval score; inspect together'),
        ('mae_cut_in / rmse_cut_in / cut_in_count', 'MWh / MWh / count', 'Descriptive station-wind stratum, not verified turbine cut-in physics'),
        ('imbalance_loss_mwh', 'weighted MWh; lower', 'Assumed asymmetric point error, not observed revenue or market settlement'),
        ('count', 'count', 'Common target samples per horizon and split'),
        ('summary suffix _mean / _sd', 'seed aggregate', 'Only SUMMARY CSV: per-seed score mean / sample SD, no seed ensemble'),
    ],columns=['metric','unit_direction','interpretation'])


def rebuild_preprocessing():
    """Rebuild in ignored scratch space, assert equality, never overwrite frozen data."""
    from src.data.load_generation import build_and_save_generation_interim
    from src.data.process_saebyeol_weather import build_and_save_master_dataset
    out=ROOT/'.experiment_archive/notebook_reproduction';out.mkdir(parents=True,exist_ok=True)
    _,generation_report=build_and_save_generation_interim(output_parquet=str(out/'generation_hourly.parquet'))
    rebuilt,_=build_and_save_master_dataset(gen_path=str(out/'generation_hourly.parquet'),
        output_path=str(out/'merged_dataset.parquet'),interim_weather_path=str(out/'weather_hourly.parquet'),max_ffill_hours=3)
    original=pd.read_parquet(ROOT/'data/processed/merged_dataset.parquet')
    pd.testing.assert_frame_equal(original,rebuilt,check_exact=True)
    return pd.DataFrame([dict(status='exact_values_match',rows=len(rebuilt),columns=len(rebuilt.columns),
        original_sha256=sha256(ROOT/'data/processed/merged_dataset.parquet'),rebuilt_sha256=sha256(out/'merged_dataset.parquet'),
        output_directory=str(out))])


def study_frame():
    f=pd.read_parquet(ROOT/'data/processed/merged_dataset.parquet')
    return f[f.datetime < '2026-01-01'].copy()


def split_quality():
    raw,_,_=prepare_splits(study_frame())
    return pd.DataFrame([dict(split=s,start=d.datetime.min(),end=d.datetime.max(),rows=len(d),
        generation_missing=int(d.generation_mwh.isna().sum()),zero_count=int(d.generation_mwh.eq(0).sum()),
        zero_pct=100*d.generation_mwh.eq(0).sum()/d.generation_mwh.notna().sum(),
        incomplete_weather_rows=int(d[WEATHER_COLUMNS].isna().any(axis=1).sum())) for s,d in raw.items()])


def window_counts():
    _,arrays,_=prepare_splits(study_frame())
    cfg=yaml.safe_load((ROOT/'configs/all_baselines.yaml').read_text())
    return pd.DataFrame([dict(split=s,horizon_hours=h,valid_windows=len(valid_window_indices(a,24,h)),
        candidate_windows=len(a)-24-h+1,excluded_windows=len(a)-24-h+1-len(valid_window_indices(a,24,h)))
        for s,a in arrays.items() for h in cfg['horizons']])


def input_example(split='validation',horizon=6):
    raw,arrays,_=prepare_splits(study_frame());i=valid_window_indices(arrays[split],24,horizon)[0]
    X=pd.DataFrame(arrays[split][i-24:i],columns=['generation_mwh']+WEATHER_COLUMNS,index=raw[split].datetime.iloc[i-24:i])
    X['generation_mwh']/=21
    X=X.rename(columns={'generation_mwh':'generation_div_21'})
    target=pd.DataFrame([dict(issue_time=raw[split].datetime.iloc[i-1],target_time=raw[split].datetime.iloc[i+horizon-1],
        horizon_hours=horizon,generation_mwh=float(arrays[split][i+horizon-1,0]))])
    return X,target


def plot_dataset():
    import matplotlib.dates as mdates
    f=study_frame();raw,_,_=prepare_splits(f)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    daily=f.set_index('datetime').generation_mwh.resample('D').mean()
    axes[0,0].plot(daily.index,daily,lw=.65);axes[0,0].set(title='Daily mean generation (2023–2025)',ylabel='MWh per hour')
    for date in ['2024-07-01','2025-01-01']:axes[0,0].axvline(pd.Timestamp(date),color='black',ls='--',lw=.8)
    for s,d in raw.items():axes[0,1].hist(d.generation_mwh.dropna(),bins=np.linspace(0,21,43),density=True,histtype='step',label=s)
    axes[0,1].set(title='Observed generation distribution',xlabel='MWh');axes[0,1].legend()
    # Only observed (not imputed) weather: association, not a turbine power curve.
    obs=f[f.generation_mwh.notna() & ~f.aws_wind_speed_is_missing]
    axes[1,0].hexbin(obs.aws_wind_speed,obs.generation_mwh,gridsize=45,mincnt=1,bins='log',cmap='viridis')
    axes[1,0].set(title='Station wind / generation association',xlabel='Station wind (m/s)',ylabel='Generation (MWh)')
    monthly=f.set_index('datetime').generation_mwh.resample('MS').apply(lambda x:x.eq(0).sum()/x.notna().sum())
    axes[1,1].plot(monthly.index,monthly,marker='o',ms=3);axes[1,1].set(title='Observed exact-zero proportion',ylabel='Fraction')
    for ax in [axes[0,0],axes[1,1]]:
        locator=mdates.AutoDateLocator(minticks=4,maxticks=7)
        ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    return fig


def weather_quality():
    f=study_frame();cols=['aws_wind_speed','aws_wind_direction','aws_temperature','aws_humidity','aws_local_pressure']
    return pd.DataFrame([dict(variable=c,original_missing=int(f[c+'_is_missing'].sum()),
        forward_filled=int(f[c+'_is_imputed'].sum()),remaining_missing=int(f[c].isna().sum())) for c in cols])


def model_catalog():
    rows=[]
    for name in ALL_MODELS:
        if name in NEURAL:
            source={'EMFN (Proposed)':'emfn.py','DLinear (+Weather)':'dlinear.py','LSTM (+Weather)':'rnn.py','GRU (+Weather)':'rnn.py','CNN-LSTM (+Weather)':'cnn_lstm.py','TCN (+Weather)':'tcn_model.py','Transformer (+Weather)':'transformer.py'}[name]
            inputs='24h generation + six weather channels';output='Bernoulli–Beta' if name==NEURAL[0] else 'point (Dirac for CRPS)'
        elif name in TREES:
            source='probabilistic_lightgbm.py';inputs='same flattened 24h × 7 inputs';output='point (Dirac for CRPS)' if name==TREES[0] else 'quantile distribution'
        elif name==LONG_TREE:
            source='lightgbm_model.py / run_all_baselines.py';inputs='up to 168h lag + weather + calendar';output='point (Dirac for CRPS)'
        else:
            source='prophet_model.py' if name=='Prophet' else 'arima_model.py' if name=='ARIMA' else 'persistence.py'
            inputs={'Prophet':'train-only trend/calendar; no rolling observations','ARIMA':'causally filtered generation history','Naive Persistence':'generation at issue time','Diurnal Persistence':'generation 24h before target'}[name]
            output='censored Gaussian' if name=='ARIMA' else 'point (Dirac for CRPS)'
        rows.append(dict(model=name,input=inputs,output=output,source='src/models/'+source))
    return pd.DataFrame(rows)


def load_run():
    path=ROOT/'reports/all_baselines_manifest.json'
    if not path.exists():raise FileNotFoundError('Run and verify: python -m src.experiments.run_all_baselines; then verify_all_baselines --run <path>')
    m=json.loads(path.read_text(encoding='utf-8'))
    v=json.loads((ROOT/'reports/all_baselines_validation.json').read_text(encoding='utf-8'))
    assert m['status']=='complete' and v['status']=='passed' and m['run_id']==v['run_id']
    assert sha256(ROOT/'data/processed/merged_dataset.parquet')==m['processed_data_sha256']
    return m,ROOT/'models/runs'/m['run_id']


def metrics_table(split='test',models=None,horizons=None,full=False):
    m,run=load_run();path=run/'metrics.csv'
    assert sha256(path)==m['artifact_hashes']['metrics.csv']
    df=pd.read_csv(path);df=df[df.split.eq(split)]
    if models is not None:df=df[df.model.isin(models)]
    if horizons is not None:df=df[df.horizon_hours.isin(horizons)]
    return df if full else df[['model','seed','horizon_hours','count']+PRIMARY]


def summary_table(split='test',models=None,horizons=None):
    df=metrics_table(split,models,horizons,full=True)
    metrics=[c for c in df.select_dtypes(include='number') if c not in ('seed','horizon_hours')]
    summary=df.groupby(['model','horizon_hours'],sort=False)[metrics].agg(['mean','std'])
    summary.columns=[f'{a}_{"sd" if b=="std" else b}' for a,b in summary.columns]
    summary.insert(0,'n_fits',df.groupby(['model','horizon_hours'],sort=False).size())
    summary=summary.reset_index()
    summary['_order']=summary.model.map({name:i for i,name in enumerate(models or ALL_MODELS)})
    return summary.sort_values(['horizon_hours','_order']).drop(columns='_order').reset_index(drop=True)


def display_scores(split='test',models=None,horizons=(1,6,24)):
    df=summary_table(split,models,horizons);out=df[['model','horizon_hours','n_fits']].copy()
    for metric in PRIMARY:
        out[metric]=[('N/A' if pd.isna(mean) else f'{mean:.4f}' if pd.isna(sd) else f'{mean:.4f} ± {sd:.4f}')
                     for mean,sd in zip(df[metric+'_mean'],df[metric+'_sd'])]
    return out


def run_status():
    m,_=load_run()
    return pd.DataFrame([dict(run_id=m['run_id'],status=m['status'],models=len({e['model'] for e in m['experiments']}),
        local_artifacts_available=(ROOT/'models/runs'/m['run_id']/'metrics.csv').exists(),
        model_horizon_seed_combinations=len(m['experiments']),metric_rows=2*len(m['experiments']),
        seeds=str(m['config']['seeds']),horizons=str(m['config']['horizons']),test_role=m['test_role'],data_sha256=m['processed_data_sha256'])])


def validation_diagnostics():
    m,_=load_run()
    return pd.DataFrame([{k:e.get(k) for k in ['model','horizon_hours','seed','selection_basis','best_epoch','stopped_epoch','best_val_crps','validation_raw_std_mwh','validation_clipped_std_mwh','validation_constant_warning']} for e in m['experiments']])


def train_all(resume=None):
    from threadpoolctl import threadpool_limits
    from src.experiments.run_all_baselines import run
    with threadpool_limits(limits=4):return run(resume)


def publish_run(run):
    """Verify a completed run, then refresh its reports and presentation tables."""
    from threadpoolctl import threadpool_limits
    from src.experiments.verify_all_baselines import verify
    from src.experiments.report_all_baselines import main as report
    from src.experiments.export_paper_views import main as export
    with threadpool_limits(limits=4):
        result=verify(run)
        report()
        export()
    return result


def train_emfn_example(horizon=1,seed=42):
    """Optional single fit with the frozen recipe, kept separate from paper results."""
    from datetime import datetime
    import torch
    from src.models.emfn_trainer import train_and_evaluate_emfn
    cfg=yaml.safe_load((ROOT/'configs/all_baselines.yaml').read_text())
    if horizon not in cfg['horizons'] or seed not in cfg['seeds']:raise ValueError('Use a registered horizon/seed')
    _,a,_=prepare_splits(study_frame());seed_everything(seed);torch.set_num_threads(cfg['torch_threads'])
    out=ROOT/'.experiment_archive/notebook_training'/datetime.now().strftime('%Y%m%dT%H%M%S');out.mkdir(parents=True)
    metrics,_,_=train_and_evaluate_emfn(a['train'],a['validation'],a['test'],lookback_steps=24,horizon=horizon,
        epochs=cfg['emfn_epochs'],patience=cfg['emfn_patience'],lr=cfg['emfn_lr'],batch_size=cfg['batch_size'],
        capacity_mwh=cfg['capacity_mwh'],device='cuda' if torch.cuda.is_available() else 'cpu',
        save_model_path=str(out/'emfn.pt'),selection_metric='crps',crps_grid_points=cfg['crps_grid_points'])
    return pd.DataFrame([dict(output=str(out),**{k:metrics[k] for k in ['best_epoch','stopped_epoch','best_val_crps']})])


def prediction_frame(model='EMFN (Proposed)',horizon=6,seed=42,split='test'):
    m,run=load_run()
    e=next(e for e in m['experiments'] if e['model']==model and e['horizon_hours']==horizon and (e['seed']==seed or e['seed'] is None))
    path=run/e[split+'_file'];assert sha256(path)==e[split+'_sha256']
    f=pd.read_parquet(path);assert f.target_time.max()<pd.Timestamp('2026-01-01')
    return f


def plot_training(model='EMFN (Proposed)',horizon=6,seed=42):
    m,run=load_run();e=next(e for e in m['experiments'] if e['model']==model and e['horizon_hours']==horizon and e['seed']==seed)
    path=run/e['checkpoint'];history=json.loads(Path(str(path)+'.history.json').read_text())
    if isinstance(history,dict):history=history['epochs']
    df=pd.DataFrame(history)
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
    ax.plot(df.epoch,df.val_crps,label='Validation CRPS');ax.axvline(e['best_epoch'],color='black',ls='--',label='Selected epoch')
    ax.set(xlabel='Epoch',ylabel='CRPS (MWh)',title=f'{model}, +{horizon}h, seed {seed}');ax.legend()
    return fig


def architecture_table():
    import torch
    model=make_neural(NEURAL[0]);rows=[]
    for name,child in model.named_children():
        rows.append(dict(module=name,type=type(child).__name__,parameters=sum(p.numel() for p in child.parameters())))
    return pd.DataFrame(rows)


def plot_architecture():
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(12,5.2),layout='constrained');ax.set(xlim=(0,12),ylim=(0,5.2));ax.axis('off')
    boxes=[(.2,2,2,1.1,'24h generation / C\n+ standardized weather'),(2.7,2,2,1.1,'Shared dilated TCN\nlast + mean pooling'),
        (5.3,3.5,2.4,1.1,'Magnitude adapter\n+ HF / AR features'),(5.3,.6,2.4,1.1,'Zero-state adapter\n+ weather / zero summaries + AR'),
        (8.2,3.5,1.5,1.1,'alpha, beta\nsoftplus + eps'),(8.2,.6,1.5,1.1,'p = sigmoid(z)\nP(Y > 0)'),
        (10.1,2,1.7,1.1,'Hurdle distribution\non [0, C]')]
    for x,y,w,h,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc='#e8f1fa',ec='#365b7e'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    for start,end in [((2.2,2.55),(2.7,2.55)),((4.7,2.7),(5.3,4)),((4.7,2.4),(5.3,1.1)),((7.7,4),(8.2,4)),((7.7,1.1),(8.2,1.1)),((9.7,4),(10.9,3.1)),((9.7,1.1),(10.9,2))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#365b7e'))
    ax.set_title('EMFN v3: bounded statistical hurdle model (not a turbine control law)',fontsize=12)
    return fig


def plot_forecast(horizon=6,seed=42,split='test'):
    from src.experiments.run_probabilistic_benchmarks import artifact_distribution
    f=prediction_frame(horizon=horizon,seed=seed,split=split)
    # Fixed first seven target days, never choose a favorable example after inspection.
    f=f[f.target_time < f.target_time.min()+pd.Timedelta(days=7)]
    m,_=load_run();d=artifact_distribution(f,m['config'])
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    ax.fill_between(f.target_time,d['lower'],d['upper'],alpha=.2,label='90% central interval')
    ax.plot(f.target_time,f.y_true,color='black',lw=.9,label='Observed')
    ax.plot(f.target_time,d['mean'],lw=.8,label='Predictive mean')
    ax.plot(f.target_time,d['median'],lw=.8,label='Predictive median')
    ax.set(title=f'EMFN +{horizon}h: first seven available {split} days, seed {seed}',ylabel='MWh');ax.legend(ncol=4)
    return fig


def plot_horizons(models=None):
    df=summary_table(models=models or MAIN,horizons=None)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for name,d in df.groupby('model',sort=False):
        d=d.sort_values('horizon_hours')
        for ax,metric in zip(axes,['crps','rmse','mae']):
            ax.errorbar(d.horizon_hours,d[metric+'_mean'],yerr=d[metric+'_sd'].fillna(0),marker='o',ms=3,label=name)
            ax.set(title=metric.upper(),xlabel='Forecast horizon (hours)',ylabel='MWh',xticks=[1,3,6,9,12,24])
    axes[0].legend(fontsize=7)
    return fig


def plot_probability_comparison():
    names=[NEURAL[0],'ARIMA',TREES[1],TREES[2]]
    df=summary_table(models=names,horizons=None)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for name,d in df.groupby('model',sort=False):
        for ax,metric in zip(axes,['zero_brier','picp_90_pct','pinaw_90_pct']):
            ax.plot(d.horizon_hours,d[metric+'_mean'],marker='o',ms=3,label=name)
            ax.set(title=metric,xlabel='Horizon (hours)',xticks=[1,3,6,9,12,24])
    axes[1].axhline(90,color='black',ls='--',lw=.7);axes[0].legend(fontsize=6.5)
    return fig


def verify_common_samples():
    m,run=load_run();records=[]
    from src.data.forecast_protocol import align_predictions
    for split in ['validation','test']:
        for h in m['config']['horizons']:
            frames={str(i):pd.read_parquet(run/e[split+'_file']) for i,e in enumerate(m['experiments']) if e['horizon_hours']==h}
            aligned=align_predictions(frames);n=len(next(iter(aligned.values())))
            assert all(len(f)==n for f in frames.values())
            records.append(dict(split=split,horizon_hours=h,model_seed_combinations=len(frames),common_count=n))
    return pd.DataFrame(records)
