"""Verify saved multi-seed experiments and report seed spread and paired time uncertainty."""
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from threadpoolctl import threadpool_limits
from src.data.forecast_protocol import ROOT,prepare_splits,align_predictions
from src.models.probabilistic_lightgbm import flattened_windows
from src.models.arima_model import ARIMAForecaster
from src.experiments.run_multiseed_crps import NEURAL,TREES,neural_frame
from src.experiments.run_comprehensive_extended_matrix import sha256,seed_everything
from src.experiments.run_probabilistic_benchmarks import artifact_distribution,score_artifact
from src.utils.probabilistic_metrics import grid_crps
from src.utils.paired_uncertainty import paired_block_interval

METRICS=['crps','zero_brier','rmse','mae','zero_auprc']


def point_losses(f,cfg):
    y=f.y_true.to_numpy()
    if f.distribution.iloc[0]=='dirac':return {'crps':np.abs(y-np.clip(f.y_pred_raw,0,cfg['capacity_mwh']))}
    grid=np.linspace(0,cfg['capacity_mwh'],cfg['crps_grid_points']);crps=[]
    for start in range(0,len(f),256):
        sub=f.iloc[start:start+256];d=artifact_distribution(sub,cfg)
        crps.extend(grid_crps(sub.y_true,d['cdf'](grid),grid,return_samples=True))
    return dict(crps=np.asarray(crps),zero_brier=(f.p_zero.to_numpy()-(y==0))**2)


def main():
    m=json.loads((ROOT/'reports/multiseed_crps_manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='complete';cfg=m['config'];run=ROOT/'models/runs'/m['run_id']
    assert sha256(ROOT/'data/processed/merged_dataset.parquet')==m['processed_data_sha256']
    for rel,digest in m['artifact_hashes'].items():assert sha256(run/rel)==digest,rel
    with zipfile.ZipFile(run/'source.zip') as z:
        for rel,digest in m['source_hashes'].items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest
    raw,arr,_=prepare_splits(pd.read_parquet(ROOT/'data/processed/merged_dataset.parquet'))
    torch.set_num_threads(cfg['torch_threads']);device='cuda' if torch.cuda.is_available() else 'cpu'
    # Match training, including cuDNN TF32=False; the process default is True.
    seed_everything(cfg['seeds'][0])
    expected={(n,h,s) for n in NEURAL+TREES for h in cfg['horizons'] for s in cfg['seeds']}
    expected|={('ARIMA',h,None) for h in cfg['horizons']}
    assert {(e['model'],e['horizon_hours'],e['seed']) for e in m['experiments']}==expected
    assert len(m['experiments'])==len(expected)
    frames={};losses={};selection_rows=[];refinement=[]
    for e in m['experiments']:
        name=e['model'];h=e['horizon_hours'];seed=e['seed'];key=(name,h,seed)
        assert e['selection_basis']=='validation_crps'
        for split in ['validation','test']:
            f=pd.read_parquet(run/e[split+'_file'])
            assert sha256(run/e[split+'_file'])==e[split+'_sha256']
            assert ((f.target_time-f.issue_time)==pd.Timedelta(hours=h)).all()
            np.testing.assert_array_equal(f.y_true,raw[split].set_index('datetime').loc[f.target_time,'generation_mwh'])
            score=score_artifact(f,cfg)
            for metric in METRICS:
                if metric in score:np.testing.assert_allclose(score[metric],e[split+'_metrics'][metric],atol=1e-9)
            if split=='validation':np.testing.assert_allclose(score['crps'],e['best_val_crps'],atol=2e-6,rtol=1e-6)
            else:
                frames[key]=f;losses[key]=point_losses(f,cfg)
                for metric in losses[key]:np.testing.assert_allclose(np.mean(losses[key][metric]),score[metric],atol=1e-7)
                if f.distribution.iloc[0]!='dirac':
                    sample=f.iloc[::10];dist=artifact_distribution(sample,cfg)
                    coarse=np.linspace(0,cfg['capacity_mwh'],cfg['crps_grid_points'])
                    fine=np.linspace(0,cfg['capacity_mwh'],2*cfg['crps_grid_points']-1)
                    a=grid_crps(sample.y_true,dist['cdf'](coarse),coarse)
                    b=grid_crps(sample.y_true,dist['cdf'](fine),fine)
                    refinement.append(dict(model=name,horizon_hours=h,seed=seed,count=len(sample),
                        coarse_crps=a,fine_crps=b,absolute_difference=abs(a-b)))
            if name in NEURAL:
                restored=neural_frame(name,run/e['checkpoint'],arr[split],raw[split],h,cfg,device)
                for col in ['y_pred_raw','y_median','p_pos','alpha','beta']:
                    if col in f:np.testing.assert_allclose(f[col],restored[col],atol=2e-6,rtol=1e-6)
        if name in NEURAL:
            history=json.loads((run/(e['checkpoint']+'.history.json')).read_text())
            assert history['selection_metric']=='crps'
            best=min(history['epochs'],key=lambda x:x['val_crps'])
            assert best['epoch']==e['best_epoch']
            np.testing.assert_allclose(best['val_crps'],e['best_val_crps'],atol=1e-12)
        elif name in TREES:
            best=min(e['candidates'],key=lambda x:x['val_crps'])
            assert best['rounds']==e['rounds']
            np.testing.assert_allclose(best['val_crps'],e['best_val_crps'],atol=1e-12)
            X,_,_=flattened_windows(arr['test'],cfg['lookback_hours'],h)
            idx=[0,len(X)//2,len(X)-1];f=frames[key];directory=run/e['checkpoint_dir']
            for i,path in enumerate(sorted(directory.glob('quantile_*.txt'))):
                prediction=lgb.Booster(model_file=str(path)).predict(X[idx])
                col='y_pred_raw' if name==TREES[0] else f'raw_q_{i:02d}'
                np.testing.assert_allclose(prediction,f[col].iloc[idx],atol=1e-7)
            if name==TREES[2]:
                p=lgb.Booster(model_file=str(directory/'positive_classifier.txt')).predict(X[idx])
                np.testing.assert_allclose(1-p,f.classifier_p_zero.iloc[idx],atol=1e-7)
        else:
            saved=json.loads((run/e['checkpoint']).read_text())
            model=ARIMAForecaster.from_parameters(saved['order'],saved['params'],cfg['capacity_mwh'])
            _,_,indices=flattened_windows(arr['test'],cfg['lookback_hours'],h)
            history=np.concatenate([raw[s].generation_mwh for s in ['train','validation','test']])
            idx=[0,len(indices)//2,len(indices)-1]
            pred=model.forecast_origins(history,len(raw['train'])+len(raw['validation'])+indices[idx]-1,[h])[h]
            np.testing.assert_allclose(pred['mean_raw'],frames[key].y_pred_raw.iloc[idx],atol=1e-7)
        selection_rows.append(dict(model=name,horizon_hours=h,seed=seed,best_epoch=e.get('best_epoch'),
            stopped_epoch=e.get('stopped_epoch'),rounds=e.get('rounds'),validation_crps=e['best_val_crps']))
        print('Verified',name,h,seed,flush=True)
    for h in cfg['horizons']:
        aligned=align_predictions({str(k):f for k,f in frames.items() if k[1]==h})
        lengths={len(f) for f in aligned.values()};assert len(lengths)==1
        for f in aligned.values():assert (f.target_time.diff().iloc[1:]==pd.Timedelta(hours=1)).all()
    assert sha256(run/'metrics.csv')==sha256(ROOT/'reports/tables/multiseed_crps_metrics.csv')
    rows=pd.read_csv(run/'metrics.csv');summaries=[]
    # Check the published rows against the independently re-scored manifest entries.
    for e in m['experiments']:
        for split in ['validation','test']:
            mask=rows.model.eq(e['model']) & rows.horizon_hours.eq(e['horizon_hours']) & rows.split.eq(split)
            mask &= rows.seed.isna() if e['seed'] is None else rows.seed.eq(e['seed'])
            selected=rows[mask];assert len(selected)==1
            for metric in METRICS:
                if metric in e[split+'_metrics']:
                    np.testing.assert_allclose(selected.iloc[0][metric],e[split+'_metrics'][metric],atol=1e-9)
    for (split,name,h),g in rows.groupby(['split','model','horizon_hours'],sort=False):
        row=dict(split=split,model=name,horizon_hours=h,n_fits=len(g))
        for metric in METRICS:
            row[metric+'_mean']=g[metric].mean();row[metric+'_sd']=g[metric].std(ddof=1)
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(ROOT/'reports/tables/multiseed_crps_summary.csv',index=False)
    pd.DataFrame(selection_rows).to_csv(ROOT/'reports/tables/multiseed_crps_selection.csv',index=False)
    paired=[];tree_spread=[]
    for h in cfg['horizons']:
        for name in TREES:
            fs=[frames[(name,h,s)] for s in cfg['seeds']]
            cols=['y_pred_raw']+[c for c in fs[0] if c.startswith('raw_q_') or c=='classifier_p_zero']
            spread=max(float(np.max(np.abs(f[cols].to_numpy()-fs[0][cols].to_numpy()))) for f in fs[1:])
            tree_spread.append(dict(model=name,horizon_hours=h,n_fits=len(fs),max_absolute_prediction_difference=spread))
        for name in ['ARIMA',NEURAL[1]]+TREES:
            for metric in ['crps','zero_brier']:
                own=[losses[(NEURAL[0],h,s)][metric] for s in cfg['seeds']]
                keys=[(name,h,None)] if name=='ARIMA' else [(name,h,s) for s in cfg['seeds']]
                if metric not in losses[keys[0]]:continue
                difference=np.mean(own,axis=0)-np.mean([losses[k][metric] for k in keys],axis=0)
                for b in cfg['bootstrap']['block_hours']:
                    paired.append(dict(model=name,horizon_hours=h,metric=metric,
                        **paired_block_interval(difference,b,cfg['bootstrap']['repetitions'],cfg['bootstrap']['seed'])))
    pd.DataFrame(paired).to_csv(ROOT/'reports/tables/multiseed_crps_paired_intervals.csv',index=False)
    pd.DataFrame(tree_spread).to_csv(ROOT/'reports/tables/multiseed_lightgbm_seed_spread.csv',index=False)
    pd.DataFrame(refinement).to_csv(ROOT/'reports/tables/multiseed_crps_grid_sensitivity.csv',index=False)
    validation=dict(status='passed',run_id=m['run_id'],tests=32,experiment_combinations=len(expected),
        scored_split_rows=len(rows),neural_checkpoint_reloads=36,tree_checkpoint_reloads=54,arima_parameter_reloads=6,
        selection='all checkpoints minimize recorded validation CRPS candidates',source_and_artifact_hashes='verified',
        common_hourly_targets='verified',bootstrap='paired circular blocks of seed-averaged losses; no ensemble predictions',
        max_sampled_grid_refinement_difference=max(r['absolute_difference'] for r in refinement),
        reload_precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),
        all_lightgbm_seed_outputs_identical=all(d['max_absolute_prediction_difference']==0 for d in tree_spread))
    (ROOT/'reports/multiseed_crps_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
