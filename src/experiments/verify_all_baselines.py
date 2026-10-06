"""Re-score, reload, verify and publish only a complete all-model run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from src.data.forecast_protocol import ROOT,prepare_splits,align_predictions,valid_window_indices
from src.models.benchmark_registry import ALL_MODELS,NEURAL,TREES,LONG_TREE
from src.models.arima_model import ARIMAForecaster
from src.models.probabilistic_lightgbm import flattened_windows
from src.experiments.run_all_baselines import neural_frame
from src.experiments.run_comprehensive_extended_matrix import sha256,seed_everything,tabular_features
from src.experiments.run_probabilistic_benchmarks import score_artifact


def verify(run):
    run=Path(run).resolve();m=json.loads((run/'manifest.json').read_text(encoding='utf-8'));cfg=m['config']
    assert m['status']=='complete'
    assert sha256(ROOT/'data/processed/merged_dataset.parquet')==m['processed_data_sha256']
    for rel,digest in m['artifact_hashes'].items():assert sha256(run/rel)==digest,rel
    with zipfile.ZipFile(run/'source.zip') as z:
        for rel,digest in m['source_hashes'].items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest,rel
    expected={(name,h,s) for name in NEURAL+TREES+[LONG_TREE,'Prophet'] for h in cfg['horizons'] for s in cfg['seeds']}
    expected|={(name,h,None) for name in ['Naive Persistence','Diurnal Persistence','ARIMA'] for h in cfg['horizons']}
    assert len(m['experiments'])==len(expected)==234
    assert {(e['model'],e['horizon_hours'],e['seed']) for e in m['experiments']}==expected
    raw,arr,_=prepare_splits(pd.read_parquet(ROOT/'data/processed/merged_dataset.parquet'))
    torch.set_num_threads(cfg['torch_threads']);seed_everything(cfg['seeds'][0])
    device='cuda' if torch.cuda.is_available() else 'cpu';C=cfg['capacity_mwh'];L=cfg['lookback_hours']
    rows=pd.read_csv(run/'metrics.csv');assert len(rows)==468
    common={};selection=[];reloaded=0
    for num,e in enumerate(m['experiments'],1):
        name,h,seed=e['model'],e['horizon_hours'],e['seed'];key=(name,h,seed)
        for split in ['validation','test']:
            path=run/e[split+'_file'];assert sha256(path)==e[split+'_sha256'];f=pd.read_parquet(path)
            assert f.target_time.max()<pd.Timestamp('2026-01-01')
            assert ((f.target_time-f.issue_time)==pd.Timedelta(hours=h)).all()
            np.testing.assert_array_equal(f.y_true,raw[split].set_index('datetime').loc[f.target_time,'generation_mwh'])
            scores=score_artifact(f,cfg)
            for metric,value in scores.items():np.testing.assert_allclose(value,e[split+'_metrics'][metric],atol=1e-9,equal_nan=True)
            mask=rows.model.eq(name)&rows.horizon_hours.eq(h)&rows.split.eq(split)
            mask &= rows.seed.isna() if seed is None else rows.seed.eq(seed)
            row=rows[mask];assert len(row)==1
            for metric,value in scores.items():np.testing.assert_allclose(value,row.iloc[0][metric],atol=1e-9,equal_nan=True)
            if split=='validation' and 'best_val_crps' in e:
                np.testing.assert_allclose(scores['crps'],e['best_val_crps'],atol=2e-6,rtol=1e-6)
            common.setdefault((split,h),{})[str(key)]=f[['target_time','issue_time','y_true','y_pred_raw']]
            if name in NEURAL:
                restored=neural_frame(name,run/e['checkpoint'],arr[split],raw[split],h,cfg,device)
                for col in ['y_pred_raw','y_median','p_pos','alpha','beta']:
                    if col in f:np.testing.assert_allclose(f[col],restored[col],atol=2e-6,rtol=1e-6)
            if split=='test':test=f
        if name in NEURAL:
            history=json.loads((run/(e['checkpoint']+'.history.json')).read_text())
            assert history['selection_metric']=='crps'
            best=min(history['epochs'],key=lambda x:x['val_crps']);assert best['epoch']==e['best_epoch']
            np.testing.assert_allclose(best['val_crps'],e['best_val_crps'],atol=1e-12);reloaded+=1
        else:
            idx=np.asarray(valid_window_indices(arr['test'],L,h));sample=np.array([0,len(idx)//2,len(idx)-1])
            if name in TREES:
                assert min(e['candidates'],key=lambda x:x['val_crps'])['rounds']==e['rounds']
                X,_,_=flattened_windows(arr['test'],L,h);directory=run/e['checkpoint_dir']
                paths=sorted(directory.glob('quantile_*.txt'));assert len(paths)==(1 if name==TREES[0] else len(cfg['lightgbm']['quantile_levels']))
                for i,path in enumerate(paths):
                    p=lgb.Booster(model_file=str(path)).predict(X[sample])
                    col='y_pred_raw' if name==TREES[0] else f'raw_q_{i:02d}'
                    np.testing.assert_allclose(p,test[col].iloc[sample],atol=1e-7)
                if name==TREES[2]:
                    p=lgb.Booster(model_file=str(directory/'positive_classifier.txt')).predict(X[sample])
                    np.testing.assert_allclose(1-p,test.classifier_p_zero.iloc[sample],atol=1e-7)
                reloaded+=1
            elif name==LONG_TREE:
                features,cols=tabular_features(raw['test'])
                p=lgb.Booster(model_file=str(run/e['checkpoint'])).predict(features[cols].iloc[idx[sample]-1])
                np.testing.assert_allclose(p,test.y_pred_raw.iloc[sample],atol=1e-7);reloaded+=1
            elif name=='ARIMA':
                saved=json.loads((run/e['checkpoint']).read_text());model=ARIMAForecaster.from_parameters(saved['order'],saved['params'],C)
                history=np.concatenate([raw[s].generation_mwh for s in ['train','validation','test']])
                p=model.forecast_origins(history,len(raw['train'])+len(raw['validation'])+idx[sample]-1,[h])[h]
                np.testing.assert_allclose(p['mean_raw'],test.y_pred_raw.iloc[sample],atol=1e-7)
                np.testing.assert_allclose(p['std_raw'],test.std_raw.iloc[sample],atol=1e-7)
                assert min(e['candidates'],key=lambda x:x['val_crps'])['order'] in e['checkpoint'];reloaded+=1
            elif name=='Prophet':
                from prophet.serialize import model_from_json
                model=model_from_json((run/e['checkpoint']).read_text())
                future=pd.DataFrame({'ds':test.target_time.iloc[sample]})
                p=model.predict(future).yhat.to_numpy()
                np.testing.assert_allclose(p,test.y_pred_raw.iloc[sample],atol=1e-7);reloaded+=1
            else:
                j=idx-1 if name=='Naive Persistence' else idx+h-1-24
                np.testing.assert_array_equal(raw['test'].generation_mwh.to_numpy()[j],test.y_pred_raw)
        selection.append({k:e.get(k) for k in ['model','horizon_hours','seed','selection_basis','best_val_crps','best_epoch','stopped_epoch','rounds','validation_raw_std_mwh','validation_clipped_std_mwh','validation_constant_warning']})
        print(f'VERIFIED {num}/234 {name} +{h}h seed={seed}',flush=True)
    coverage=[]
    for (split,h),frames in common.items():
        aligned=align_predictions(frames);n=len(next(iter(aligned.values())))
        assert all(len(f)==n for f in frames.values())
        coverage.append(dict(split=split,horizon_hours=h,common_count=n,model_seed_combinations=len(frames)))
    metrics=[c for c in rows.select_dtypes(include='number') if c not in ['seed','horizon_hours']]
    summaries=[]
    for (split,name,h),g in rows.groupby(['split','model','horizon_hours'],sort=False):
        r=dict(split=split,model=name,horizon_hours=h,n_fits=len(g))
        for metric in metrics:r[metric+'_mean']=g[metric].mean();r[metric+'_sd']=g[metric].std(ddof=1)
        summaries.append(r)
    seed_spread=[]
    for name in TREES+[LONG_TREE,'Prophet']:
        for h in cfg['horizons']:
            entries=[e for e in m['experiments'] if e['model']==name and e['horizon_hours']==h]
            fs=[pd.read_parquet(run/e['test_file']) for e in entries]
            cols=['y_pred_raw']+[c for c in fs[0] if c.startswith('raw_q_') or c=='classifier_p_zero']
            difference=max(float(np.max(np.abs(f[cols].to_numpy()-fs[0][cols].to_numpy()))) for f in fs[1:])
            seed_spread.append(dict(model=name,horizon_hours=h,n_fits=len(fs),max_absolute_prediction_difference=difference))
    tables=ROOT/'reports/tables';outputs={}
    for name,df in [('all_baselines_metrics',rows),('all_baselines_summary',pd.DataFrame(summaries)),('all_baselines_selection',pd.DataFrame(selection)),('all_baselines_coverage',pd.DataFrame(coverage)),('all_baselines_seed_spread',pd.DataFrame(seed_spread))]:
        path=tables/(name+'.csv');df.to_csv(path,index=False);outputs[path.relative_to(ROOT).as_posix()]=sha256(path)
    result=dict(status='passed',run_id=m['run_id'],model_families=len(ALL_MODELS),experiment_combinations=234,scored_split_rows=468,
        reloaded_model_horizon_seed_combinations=reloaded,neural_full_validation_test_reloads=126,
        non_neural_reload_scope='first/middle/last test predictions per combination; persistence full test vectors',
        source_artifact_data_hashes='verified',all_metrics='recomputed',common_target_issue_labels='verified',forecast_2026_used=False,
        validation_constant_warnings=[e for e in selection if e['validation_constant_warning']],artifact_hashes=outputs,
        verifier_sha256=sha256(__file__))
    (ROOT/'reports/all_baselines_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    shutil.copy2(run/'manifest.json',ROOT/'reports/all_baselines_manifest.json')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    with threadpool_limits(limits=4):verify(args.run)
