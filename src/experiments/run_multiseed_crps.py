"""Fixed three-seed paper comparison with validation-CRPS checkpoint selection."""
import argparse
from datetime import datetime, timezone, timedelta
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
from src.data.forecast_protocol import ROOT,prepare_splits,prediction_frame,align_predictions
from src.models.emfn import EMFN
from src.models.emfn_trainer import EMFNMultiChannelDataset,train_and_evaluate_emfn
from src.models.torch_trainer import WindTimeSeriesDataset,train_and_evaluate_torch_model
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.probabilistic_lightgbm import flattened_windows,ProbabilisticLightGBM
from src.utils.probabilistic_metrics import grid_crps,quantile_distribution
from src.experiments.run_probabilistic_benchmarks import score_artifact,attach_distribution
from src.experiments.run_comprehensive_extended_matrix import sha256,slug,seed_everything,add_observations

NEURAL=['EMFN (Proposed)','CNN-LSTM (+Weather)']
TREES=['LightGBM (24h matched)','Quantile LightGBM (24h matched)','Hurdle Quantile LightGBM (24h matched)']


def neural_frame(name,checkpoint,array,raw,h,cfg,device):
    C=cfg['capacity_mwh'];L=cfg['lookback_hours']
    if name==NEURAL[0]:
        model=EMFN(lookback_len=L,pred_len=1,n_weather_features=6,capacity_mwh=C)
        ds=EMFNMultiChannelDataset(array,L,h)
    else:
        model=CNNLSTMForecaster(input_dim=7,lookback_steps=L,forecast_horizon=1)
        ds=WindTimeSeriesDataset(array,L,h,C)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.to(device).eval()
    pieces=[]
    with torch.no_grad():
        for batch in DataLoader(ds,batch_size=cfg['batch_size'],shuffle=False):
            if name==NEURAL[0]:
                pieces.append(model.predict_point_forecasts(batch[0].to(device),batch[1].to(device)))
            else:pieces.append(model(batch[0].to(device)).cpu().numpy().ravel()*C)
    y=array[np.asarray(ds.valid_indices)+h-1,0]
    if name==NEURAL[0]:
        p={k:np.concatenate([d[k].ravel() for d in pieces]) for k in pieces[0]}
        f=prediction_frame(raw,ds.valid_indices,h,y,p['y_mean_rmse'],y_median=p['y_median_mae'],
            p_pos=p['p_positive'],p_zero=p['p_zero'],alpha=p['alpha'],beta=p['beta'])
        f['distribution']='hurdle_beta'
    else:
        f=prediction_frame(raw,ds.valid_indices,h,y,np.concatenate(pieces));f['distribution']='dirac'
    return add_observations(f,raw,C)


def run(resume=None):
    cfg=yaml.safe_load((ROOT/'configs/multiseed_crps.yaml').read_text(encoding='utf-8'))
    parent=json.loads((ROOT/'reports/probabilistic_run_manifest.json').read_text(encoding='utf-8'))
    checked=json.loads((ROOT/'reports/probabilistic_artifact_validation.json').read_text(encoding='utf-8'))
    assert parent['status']=='complete' and checked['status']=='passed' and checked['run_id']==parent['run_id']
    cfg['lightgbm']=parent['config']['lightgbm']
    old=ROOT/'models/runs'/parent['run_id']
    datafile=ROOT/'data/processed/merged_dataset.parquet'
    assert sha256(datafile)==parent['processed_data_sha256']
    if resume:
        out=Path(resume).resolve()
        if out.parent!=(ROOT/'models/runs').resolve():raise ValueError('Resume must name a project run directory')
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['config']==cfg and manifest['processed_data_sha256']==sha256(datafile)
        for rel,digest in manifest['source_hashes'].items():
            if sha256(ROOT/rel)!=digest:raise ValueError('Resume source changed: '+rel)
    else:
        run_id='crps_multiseed_'+datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%dT%H%M%S%z')
        out=ROOT/'models/runs'/run_id;out.mkdir(parents=True)
        manifest=dict(run_id=run_id,status='running',parent_run_id=parent['run_id'],config=cfg,
            processed_data_sha256=sha256(datafile),test_role=cfg['test_role'],experiments=[],source_hashes={},
            environment={p:importlib.metadata.version(p) for p in ['numpy','scipy','torch','lightgbm','pandas','statsmodels']})
        files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').glob('*.yaml'))+list(ROOT.glob('test_*.py'))+[ROOT/'requirements_probabilistic.txt']
        with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in files:
                rel=p.relative_to(ROOT).as_posix();manifest['source_hashes'][rel]=sha256(p);z.write(p,rel)
        (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    save();print('RUN',out.name,flush=True)
    torch.set_num_threads(cfg['torch_threads'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest.update(device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None)
    raw,arrays,scaler=prepare_splits(pd.read_parquet(datafile));manifest['weather_scaler']=scaler
    C=cfg['capacity_mwh'];L=cfg['lookback_hours'];grid=np.linspace(0,C,cfg['crps_grid_points'])
    def done(name,h,seed):return any(e['model']==name and e['horizon_hours']==h and e['seed']==seed for e in manifest['experiments'])
    def record(name,h,seed,frames,info):
        tag=f'{slug(name)}_{h}h_seed{seed}'
        entry=dict(model=name,horizon_hours=h,seed=seed,selection_basis='validation_crps',**info)
        for split,f in frames.items():
            f.to_parquet(out/f'{tag}_{split}.parquet',index=False)
            entry[split+'_file']=f'{tag}_{split}.parquet'
            entry[split+'_sha256']=sha256(out/entry[split+'_file'])
        entry['validation_metrics']=score_artifact(frames['validation'],cfg)
        # The chosen checkpoint must reproduce the score used for selection.
        np.testing.assert_allclose(entry['validation_metrics']['crps'],info['best_val_crps'],atol=2e-6,rtol=1e-6)
        entry['test_metrics']=score_artifact(frames['test'],cfg)
        ref=pd.read_parquet(old/f'{slug(NEURAL[0])}_{h}h_test.parquet')
        aligned=align_predictions({'parent':ref,'new':frames['test']})
        assert len(aligned['new'])==len(ref)==len(frames['test'])
        manifest['experiments'].append(entry);save()
        print(f'DONE {len(manifest["experiments"])}/96 | seed={seed} +{h}h {name} | validation CRPS={info["best_val_crps"]:.6f}',flush=True)
    # ARIMA has no random seed. Reuse the verified deterministic fit, once per horizon.
    for h in cfg['horizons']:
        if done('ARIMA',h,None):continue
        selected=next(e for e in parent['selections'] if e['model']=='ARIMA' and e['horizon_hours']==h)
        checkpoint=selected['checkpoint'];assert sha256(old/checkpoint)==parent['artifact_hashes'][checkpoint]
        shutil.copy2(old/checkpoint,out/checkpoint)
        frames={}
        for split in ['validation','test']:
            file=f'arima_{h}h_{split}.parquet';assert sha256(old/file)==parent['artifact_hashes'][file]
            frames[split]=pd.read_parquet(old/file)
        record('ARIMA',h,None,frames,dict(best_val_crps=selected['val_crps'],checkpoint=checkpoint,
            reuse_reason='deterministic_fit_already_selected_by_validation_CRPS',source_run_id=parent['run_id']))
    for seed in cfg['seeds']:
        for h in cfg['horizons']:
            for name in NEURAL:
                if done(name,h,seed):continue
                seed_everything(seed);checkpoint=f'{slug(name)}_{h}h_seed{seed}.pt';start=time.perf_counter()
                print(f'TRAIN seed={seed} +{h}h {name}',flush=True)
                common=dict(lookback_steps=L,horizon=h,batch_size=cfg['batch_size'],device=device,
                    save_model_path=str(out/checkpoint),selection_metric='crps')
                if name==NEURAL[0]:
                    info,pred,_=train_and_evaluate_emfn(arrays['train'],arrays['validation'],arrays['test'],
                        epochs=cfg['emfn_epochs'],patience=cfg['emfn_patience'],lr=cfg['emfn_lr'],
                        capacity_mwh=C,crps_grid_points=cfg['crps_grid_points'],**common)
                else:
                    model=CNNLSTMForecaster(input_dim=7,lookback_steps=L,forecast_horizon=1)
                    info,_,pred=train_and_evaluate_torch_model(model,arrays['train'],arrays['validation'],arrays['test'],
                        epochs=cfg['cnn_lstm_epochs'],lr=cfg['cnn_lstm_lr'],rated_capacity_mwh=C,
                        patience=cfg['cnn_lstm_patience'],**common)
                frames={s:neural_frame(name,out/checkpoint,arrays[s],raw[s],h,cfg,device) for s in ['validation','test']}
                np.testing.assert_allclose(frames['test'].y_pred_raw,pred,atol=2e-6,rtol=1e-6)
                record(name,h,seed,frames,dict(checkpoint=checkpoint,train_time_sec=time.perf_counter()-start,
                    **{k:info[k] for k in ['best_epoch','stopped_epoch','best_val_loss','best_val_crps','train_count','val_count']}))
            Xtr,ytr,_=flattened_windows(arrays['train'],L,h)
            inputs={s:flattened_windows(arrays[s],L,h) for s in ['validation','test']}
            for name in TREES:
                if done(name,h,seed):continue
                point=name==TREES[0];hurdle=name==TREES[2];start=time.perf_counter()
                model=ProbabilisticLightGBM(cfg['lightgbm'],seed,cfg['torch_threads'],point=point,hurdle=hurdle).fit(Xtr,ytr)
                candidates=[];Xv,yv,_=inputs['validation']
                for rounds in cfg['lightgbm']['candidate_rounds']:
                    if point:
                        yp,_=model.predict(Xv,rounds);score=float(np.abs(yv-np.clip(yp[:,0],0,C)).mean())
                    else:score=grid_crps(yv,model.distribution(Xv,rounds,C)['cdf'](grid),grid)
                    candidates.append(dict(rounds=rounds,val_crps=score))
                best=min(candidates,key=lambda c:c['val_crps']);rounds=best['rounds']
                directory=f'{slug(name)}_{h}h_seed{seed}_models';model.save(out/directory,rounds)
                frames={}
                for s,(X,y,idx) in inputs.items():
                    f=add_observations(prediction_frame(raw[s],idx,h,y,np.zeros(len(y))),raw[s],C)
                    yp,p0=model.predict(X,rounds)
                    if point:
                        f['distribution']='dirac';f['y_pred_raw']=yp[:,0];f['y_pred_clipped']=np.clip(yp[:,0],0,C)
                    else:
                        f=attach_distribution(f,quantile_distribution(yp,model.levels,C,p0,cfg['lightgbm']['positive_quantile_floor_mwh']),
                            'hurdle_quantile' if hurdle else 'quantile')
                        for i in range(yp.shape[1]):f[f'raw_q_{i:02d}']=yp[:,i]
                        if hurdle:f['classifier_p_zero']=p0
                    frames[s]=f
                record(name,h,seed,frames,dict(checkpoint_dir=directory,best_val_crps=best['val_crps'],
                    candidates=candidates,rounds=rounds,train_time_sec=time.perf_counter()-start))
    assert len(manifest['experiments'])==6+len(cfg['seeds'])*len(cfg['horizons'])*5
    rows=[]
    for e in manifest['experiments']:
        for split in ['validation','test']:
            rows.append(dict(model=e['model'],seed=e['seed'],horizon_hours=e['horizon_hours'],split=split,
                selection_basis=e['selection_basis'],**e[split+'_metrics']))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    manifest['status']='complete'
    manifest['artifact_hashes']={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    save();shutil.copy2(out/'manifest.json',ROOT/'reports/multiseed_crps_manifest.json')
    shutil.copy2(out/'metrics.csv',ROOT/'reports/tables/multiseed_crps_metrics.csv')
    print('COMPLETE',out,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume');args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.resume)
