"""ARIMA and matched-input probabilistic LightGBM extension; preserves prior results."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, json, shutil, warnings, zipfile, importlib.metadata
import numpy as np
import pandas as pd
import yaml
from scipy.stats import beta as beta_dist
from threadpoolctl import threadpool_limits
from src.data.forecast_protocol import ROOT,prepare_splits,prediction_frame,align_predictions,load_config
from src.models.arima_model import ARIMAForecaster
from src.models.probabilistic_lightgbm import flattened_windows,ProbabilisticLightGBM
from src.utils.probabilistic_metrics import quantile_distribution,censored_normal_distribution,score_distribution,grid_crps
from src.utils.metrics import evaluate_wind_forecast
from src.experiments.run_comprehensive_extended_matrix import sha256,slug,add_observations


def artifact_distribution(f, cfg):
    kind=f.distribution.iloc[0];C=cfg['capacity_mwh']
    if kind=='censored_normal':return censored_normal_distribution(f.y_pred_raw,f.std_raw,C)
    if kind in ('quantile','hurdle_quantile'):
        cols=[f'raw_q_{i:02d}' for i in range(len(cfg['lightgbm']['quantile_levels']))]
        p0=f.classifier_p_zero.to_numpy() if kind=='hurdle_quantile' else None
        return quantile_distribution(f[cols].to_numpy(),cfg['lightgbm']['quantile_levels'],C,p0,cfg['lightgbm']['positive_quantile_floor_mwh'])
    if kind=='hurdle_beta':
        p=f.p_pos.to_numpy(dtype=float);a=f.alpha.to_numpy(dtype=float);b=f.beta.to_numpy(dtype=float)
        from src.utils.metrics import hurdle_beta_quantiles
        return dict(mean=f.y_pred_raw.to_numpy(),median=f.y_median.to_numpy(),raw_mean=f.y_pred_raw.to_numpy(),p_zero=1-p,
            lower=hurdle_beta_quantiles(p,a,b,.05,C),upper=hurdle_beta_quantiles(p,a,b,.95,C),
            cdf=lambda grid:1-p[:,None]+p[:,None]*beta_dist.cdf(np.asarray(grid)[None,:]/C,a[:,None],b[:,None]))
    raise ValueError(kind)


def score_artifact(f,cfg):
    if f.distribution.iloc[0]=='dirac':
        m=evaluate_wind_forecast(f.y_true,f.y_pred_raw,y_pred_raw=f.y_pred_raw,
            wind_speed=f.wind_speed_target,rated_capacity_mwh=cfg['capacity_mwh'],timestamps=f.target_time)
        m['mae_mean']=m['mae'];m['crps']=float(np.abs(f.y_true-np.clip(f.y_pred_raw,0,cfg['capacity_mwh'])).mean())
        return m
    return score_distribution(f,artifact_distribution(f,cfg),cfg['capacity_mwh'],cfg['crps_grid_points'])


def attach_distribution(f,d,kind):
    f=f.copy();f['distribution']=kind;f['y_pred_raw']=d['raw_mean'];f['y_pred_mean']=d['mean']
    f['y_median']=d['median'];f['p_zero']=d['p_zero'];f['lower_90']=d['lower'];f['upper_90']=d['upper']
    # This is the mean of the bounded distribution, not clip(raw mean).
    f['y_pred_clipped']=d['mean']
    return f


def run():
    cfg=yaml.safe_load((ROOT/'configs/probabilistic_benchmarks.yaml').read_text(encoding='utf-8-sig'))
    parent=json.loads((ROOT/'reports/corrected_run_manifest.json').read_text(encoding='utf-8'))
    oldrun=ROOT/'models/runs'/parent['run_id']
    datafile=ROOT/'data/processed/merged_dataset.parquet'
    if sha256(datafile)!=parent['processed_data_sha256']:raise ValueError('Data differ from corrected parent run')
    run_id='probabilistic_extension_'+datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%dT%H%M%S%z')
    out=ROOT/'models/runs'/run_id;out.mkdir(parents=True)
    manifest=dict(run_id=run_id,parent_run_id=parent['run_id'],status='running',config=cfg,
        processed_data_sha256=sha256(datafile),test_role=parent['test_role'],
        environment={p:importlib.metadata.version(p) for p in ['numpy','scipy','statsmodels','lightgbm','pandas','torch']},
        selections=[],arima_fits=[],source_hashes={})
    source=list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').glob('*.yaml'))+[ROOT/'test_probabilistic_extension.py',ROOT/'requirements_probabilistic.txt']
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in source:
            rel=p.relative_to(ROOT).as_posix();manifest['source_hashes'][rel]=sha256(p);z.write(p,rel)
    shutil.copy2(ROOT/'configs/probabilistic_benchmarks.yaml',out/'config.yaml')
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    save();print('RUN',run_id,flush=True)
    raw,arrays,scaler=prepare_splits(pd.read_parquet(datafile))
    manifest['weather_scaler']=scaler
    C=cfg['capacity_mwh'];L=cfg['lookback_hours'];grid=np.linspace(0,C,cfg['crps_grid_points'])
    train_y=raw['train'].generation_mwh.to_numpy()
    val_history=np.r_[train_y,raw['validation'].generation_mwh.to_numpy()]
    all_y=np.r_[val_history,raw['test'].generation_mwh.to_numpy()]
    val_origins=np.arange(len(train_y)+L-1,len(val_history)-1)
    arimas={};arima_val={}
    for order in cfg['arima_orders']:
        label=','.join(map(str,order))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model=ARIMAForecaster(order,C,cfg['arima_maxiter']).fit(train_y)
        record=dict(order=order,converged=model.converged,params=model.params.tolist(),
            warnings=[str(w.message) for w in caught])
        manifest['arima_fits'].append(record)
        (out/f'arima_{label}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        if model.converged:
            arimas[label]=model;arima_val[label]=model.forecast_origins(val_history,val_origins,cfg['horizons'])
        print('ARIMA fitted',order,'converged',model.converged,flush=True);save()
    if not arimas:raise RuntimeError('No ARIMA candidate converged')
    rows=[];val_rows=[];candidate_rows=[];coverage=[]
    for h in cfg['horizons']:
        Xtr,ytr,_=flattened_windows(arrays['train'],L,h)
        Xval,yval,ival=flattened_windows(arrays['validation'],L,h)
        Xtest,ytest,itest=flattened_windows(arrays['test'],L,h)
        val_ref=add_observations(prediction_frame(raw['validation'],ival,h,yval,np.zeros(len(yval))),raw['validation'],C)
        test_ref=add_observations(prediction_frame(raw['test'],itest,h,ytest,np.zeros(len(ytest))),raw['test'],C)
        frames={}
        for e in [e for e in parent['experiments'] if e['horizon']==h]:
            name=e['model'];f=pd.read_parquet(oldrun/f'{slug(name)}_{h}h_common.parquet')
            f['distribution']='hurdle_beta' if name=='EMFN (Proposed)' else 'dirac';frames[name]=f
        best=None
        for label in arimas:
            raw_pred=arima_val[label][h]
            distribution=censored_normal_distribution(raw_pred['mean_raw'][ival-L],raw_pred['std_raw'][ival-L],C)
            crps=grid_crps(val_ref.y_true,distribution['cdf'](grid),grid)
            candidate_rows.append(dict(model='ARIMA',horizon_hours=h,candidate=label,val_crps=crps,count=len(ival)))
            if best is None or crps<best[0]:best=(crps,label,distribution)
        best_crps,label,dval=best
        model=arimas[label]
        val_f=attach_distribution(val_ref,dval,'censored_normal')
        val_f['std_raw']=arima_val[label][h]['std_raw'][ival-L]
        arima_test=model.forecast_origins(all_y,len(val_history)+itest-1,[h])[h]
        dtest=censored_normal_distribution(arima_test['mean_raw'],arima_test['std_raw'],C)
        test_f=attach_distribution(test_ref,dtest,'censored_normal');test_f['std_raw']=arima_test['std_raw']
        frames['ARIMA']=test_f
        val_f.to_parquet(out/f'arima_{h}h_validation.parquet',index=False)
        manifest['selections'].append(dict(model='ARIMA',horizon_hours=h,order=model.order,val_crps=best_crps,checkpoint=f'arima_{label}.json'))
        vals=score_artifact(val_f,cfg);vals.update(model='ARIMA',horizon_hours=h,run_id=run_id);val_rows.append(vals)
        print(f'+{h}h ARIMA order {label} selected on validation CRPS={best_crps:.5f}',flush=True)
        for name,point,hurdle in [('LightGBM (24h matched)',True,False),('Quantile LightGBM (24h matched)',False,False),('Hurdle Quantile LightGBM (24h matched)',False,True)]:
            model=ProbabilisticLightGBM(cfg['lightgbm'],cfg['seed'],cfg['threads'],hurdle=hurdle,point=point).fit(Xtr,ytr)
            best=None
            for rounds in cfg['lightgbm']['candidate_rounds']:
                if point:
                    rawpred,_=model.predict(Xval,rounds);value=float(np.abs(val_ref.y_true-np.clip(rawpred[:,0],0,C)).mean())
                else:
                    distribution=model.distribution(Xval,rounds,C);value=grid_crps(val_ref.y_true,distribution['cdf'](grid),grid)
                candidate_rows.append(dict(model=name,horizon_hours=h,candidate=str(rounds),val_crps=value,count=len(ival)))
                if best is None or value<best[0]:best=(value,rounds)
            value,rounds=best
            directory=out/f'{slug(name)}_{h}h_models';model.save(directory,rounds)
            manifest['selections'].append(dict(model=name,horizon_hours=h,rounds=rounds,val_crps=value,checkpoint_dir=directory.name))
            for split,X,ref in [('validation',Xval,val_ref),('test',Xtest,test_ref)]:
                rawpred,p0=model.predict(X,rounds)
                f=ref.copy()
                if point:
                    f['distribution']='dirac';f['y_pred_raw']=rawpred[:,0];f['y_pred_clipped']=np.clip(rawpred[:,0],0,C)
                else:
                    d=quantile_distribution(rawpred,model.levels,C,p0,cfg['lightgbm']['positive_quantile_floor_mwh'])
                    f=attach_distribution(f,d,'hurdle_quantile' if hurdle else 'quantile')
                    for i in range(rawpred.shape[1]):f[f'raw_q_{i:02d}']=rawpred[:,i]
                    if hurdle:f['classifier_p_zero']=p0
                if split=='test':frames[name]=f
                else:
                    f.to_parquet(out/f'{slug(name)}_{h}h_validation.parquet',index=False)
                    vals=score_artifact(f,cfg);vals.update(model=name,horizon_hours=h,run_id=run_id);val_rows.append(vals)
            print(f'+{h}h {name}: rounds={rounds}, validation CRPS={value:.5f}',flush=True);save()
        aligned=align_predictions(frames)
        if len(aligned['EMFN (Proposed)'])!=len(frames['EMFN (Proposed)']):raise ValueError('New model changed common parent test samples')
        for name,f in aligned.items():
            f.to_parquet(out/f'{slug(name)}_{h}h_test.parquet',index=False)
            m=score_artifact(f,cfg)
            m.update(model=name,horizon_hours=h,horizon=f'+{h}h',run_id=run_id,seed=cfg['seed'],
                selection_basis='validation_crps' if name in ['ARIMA','LightGBM (24h matched)','Quantile LightGBM (24h matched)','Hurdle Quantile LightGBM (24h matched)'] else 'parent_run_legacy_selection',
                distribution=f.distribution.iloc[0])
            rows.append(m);coverage.append(dict(model=name,horizon_hours=h,count=len(f),parent_common_count=len(frames['EMFN (Proposed)'])))
        pd.DataFrame(rows).to_csv(out/'benchmark.csv',index=False)
        pd.DataFrame(val_rows).to_csv(out/'validation_selected.csv',index=False)
        pd.DataFrame(candidate_rows).to_csv(out/'validation_candidates.csv',index=False)
        save()
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    manifest['status']='complete';manifest['artifact_hashes']={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    save()
    tables=ROOT/'reports/tables'
    for source,target in [('benchmark.csv','probabilistic_extended_benchmark.csv'),('validation_selected.csv','probabilistic_validation_selected.csv'),('validation_candidates.csv','probabilistic_validation_candidates.csv'),('coverage.csv','probabilistic_benchmark_coverage.csv')]:shutil.copy2(out/source,tables/target)
    all_results=pd.DataFrame(rows);paper=all_results[all_results.model.isin(cfg['paper_shortlist']['models']+['EMFN (Proposed)'])]
    paper[['model','horizon','count','crps','zero_brier','rmse','mae','zero_auprc','selection_basis']].to_csv(tables/'paper_five_comparators.csv',index=False)
    shutil.copy2(out/'manifest.json',ROOT/'reports/probabilistic_run_manifest.json')
    print('COMPLETE',out,flush=True)
    return out

if __name__=='__main__':
    with threadpool_limits(limits=4):run()
