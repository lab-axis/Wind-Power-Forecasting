"""Fresh all-model benchmark. Run with -m; resumable, never scores 2026.

Training/selection stays in Python. Notebooks call this entry point and inspect
its immutable per-run checkpoints, timestamps, predictions, and provenance.
"""
import argparse
from datetime import datetime, timezone, timedelta
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import warnings
import zipfile
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
from src.data.forecast_protocol import ROOT, prepare_splits, prediction_frame, align_predictions, valid_window_indices
from src.models.benchmark_registry import NEURAL, TREES, LONG_TREE, ALL_MODELS, make_neural
from src.models.emfn_trainer import EMFNMultiChannelDataset, train_and_evaluate_emfn
from src.models.torch_trainer import WindTimeSeriesDataset, train_and_evaluate_torch_model
from src.models.probabilistic_lightgbm import flattened_windows, ProbabilisticLightGBM
from src.models.arima_model import ARIMAForecaster
from src.utils.probabilistic_metrics import grid_crps, quantile_distribution, censored_normal_distribution
from src.experiments.run_probabilistic_benchmarks import score_artifact, attach_distribution
from src.experiments.run_comprehensive_extended_matrix import sha256, slug, seed_everything, add_observations, tabular_features


def neural_frame(name, checkpoint, array, raw, h, cfg, device):
    C, L = cfg['capacity_mwh'], cfg['lookback_hours']
    model = make_neural(name, L, C)
    ds = EMFNMultiChannelDataset(array, L, h) if name == NEURAL[0] else WindTimeSeriesDataset(array, L, h, C)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.to(device).eval()
    pieces = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False):
            if name == NEURAL[0]:
                pieces.append(model.predict_point_forecasts(batch[0].to(device), batch[1].to(device)))
            else:
                pieces.append(model(batch[0].to(device)).cpu().numpy().ravel()*C)
    y = array[np.asarray(ds.valid_indices)+h-1, 0]
    if name == NEURAL[0]:
        p = {k: np.concatenate([d[k].ravel() for d in pieces]) for k in pieces[0]}
        f = prediction_frame(raw, ds.valid_indices, h, y, p['y_mean_rmse'], y_median=p['y_median_mae'],
                             p_pos=p['p_positive'], p_zero=p['p_zero'], alpha=p['alpha'], beta=p['beta'])
        f['distribution'] = 'hurdle_beta'
    else:
        f = prediction_frame(raw, ds.valid_indices, h, y, np.concatenate(pieces))
        f['distribution'] = 'dirac'
    return add_observations(f, raw, C)


def run(resume=None):
    cfg = yaml.safe_load((ROOT/'configs/all_baselines.yaml').read_text(encoding='utf-8'))
    datafile = ROOT/'data/processed/merged_dataset.parquet'
    if resume:
        out = Path(resume).resolve()
        if out.parent != (ROOT/'models/runs').resolve():
            raise ValueError('Resume must name a project run directory')
        manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['config'] == cfg and manifest['processed_data_sha256'] == sha256(datafile)
        for rel, digest in manifest['source_hashes'].items():
            if sha256(ROOT/rel) != digest:
                raise ValueError('Resume source changed: '+rel)
    else:
        run_id = 'all_baselines_'+datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%dT%H%M%S%z')
        out = ROOT/'models/runs'/run_id
        out.mkdir(parents=True)
        manifest = dict(run_id=run_id, status='running', config=cfg, experiments=[], arima_fits=[],
                        processed_data_sha256=sha256(datafile), test_role=cfg['test_role'], source_hashes={},
                        raw_hashes={p.relative_to(ROOT).as_posix(): sha256(p) for p in (ROOT/'data/raw').rglob('*')
                                    if p.is_file() and p.suffix.lower() in ('.csv', '.xlsx', '.xls') and 'minute' not in str(p)},
                        environment={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'torch', 'lightgbm', 'pandas', 'statsmodels', 'prophet']})
        # Freeze dependencies used by this run; notebook/report-only additions may follow.
        files = list((ROOT/'src/models').rglob('*.py')) + list((ROOT/'src/data').glob('*.py')) + list((ROOT/'src/features').glob('*.py')) + list((ROOT/'src/utils').glob('*.py'))
        files += [Path(__file__), ROOT/'src/experiments/run_probabilistic_benchmarks.py', ROOT/'src/experiments/run_comprehensive_extended_matrix.py', ROOT/'configs/all_baselines.yaml', ROOT/'configs/base_config.yaml']
        with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in files:
                rel = p.relative_to(ROOT).as_posix()
                manifest['source_hashes'][rel] = sha256(p)
                z.write(p, rel)
        (out/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    save()
    print('RUN', out.name, flush=True)
    torch.set_num_threads(cfg['torch_threads'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    manifest.update(device=device, gpu=torch.cuda.get_device_name(0) if device=='cuda' else None)
    raw, arrays, scaler = prepare_splits(pd.read_parquet(datafile))
    assert max(d.datetime.max() for d in raw.values()) < pd.Timestamp('2026-01-01')
    manifest['weather_scaler'] = scaler
    C, L = cfg['capacity_mwh'], cfg['lookback_hours']
    grid = np.linspace(0, C, cfg['crps_grid_points'])
    indices = {(s,h): np.asarray(valid_window_indices(arrays[s], L, h)) for s in raw for h in cfg['horizons']}
    def reference(s,h):
        idx = indices[s,h]
        return add_observations(prediction_frame(raw[s], idx, h, arrays[s][idx+h-1,0], np.zeros(len(idx))), raw[s], C)
    def done(name,h,seed):
        return any(e['model']==name and e['horizon_hours']==h and e['seed']==seed for e in manifest['experiments'])
    def record(name,h,seed,frames,info,basis='validation_crps'):
        tag = f'{slug(name)}_{h}h_seed{seed}'
        entry = dict(model=name, horizon_hours=h, seed=seed, selection_basis=basis, **info)
        for s,f in frames.items():
            aligned = align_predictions({'reference': reference(s,h), 'model': f})
            assert len(aligned['model']) == len(f) == len(reference(s,h))
            f.to_parquet(out/f'{tag}_{s}.parquet', index=False)
            entry[s+'_file'] = f'{tag}_{s}.parquet'
            entry[s+'_sha256'] = sha256(out/entry[s+'_file'])
            entry[s+'_metrics'] = score_artifact(f,cfg)
        if 'best_val_crps' in info:
            np.testing.assert_allclose(entry['validation_metrics']['crps'], info['best_val_crps'], atol=2e-6, rtol=1e-6)
        entry['validation_raw_std_mwh'] = float(frames['validation'].y_pred_raw.std(ddof=0))
        entry['validation_clipped_std_mwh'] = float(frames['validation'].y_pred_clipped.std(ddof=0))
        entry['validation_constant_warning'] = entry['validation_clipped_std_mwh'] < 1e-6
        manifest['experiments'].append(entry)
        save()
        print(f'DONE {len(manifest["experiments"])}/234 | seed={seed} +{h}h {name} | validation CRPS={entry["validation_metrics"]["crps"]:.6f}', flush=True)

    # Persistence is recomputed directly at each common forecast issue timestamp.
    for h in cfg['horizons']:
        for name in ['Naive Persistence', 'Diurnal Persistence']:
            if done(name,h,None): continue
            frames = {}
            for s in ['validation','test']:
                idx = indices[s,h]
                j = idx-1 if name=='Naive Persistence' else idx+h-1-24
                f = reference(s,h)
                f['y_pred_raw'] = raw[s].generation_mwh.to_numpy()[j]
                f['y_pred_clipped'] = np.clip(f.y_pred_raw,0,C)
                f['distribution'] = 'dirac'
                frames[s] = f
            record(name,h,None,frames,{},'fixed_rule_no_selection')

    train_y = raw['train'].generation_mwh.to_numpy()
    all_y = np.concatenate([raw[s].generation_mwh.to_numpy() for s in ['train','validation','test']])
    offsets = {'validation': len(train_y), 'test': len(train_y)+len(raw['validation'])}
    arimas = {}
    if not all(done('ARIMA',h,None) for h in cfg['horizons']):
        for order in cfg['arima_orders']:
            label = ','.join(map(str,order)); file = out/f'arima_{label}.json'
            if file.exists():
                fit = json.loads(file.read_text()); model = ARIMAForecaster.from_parameters(order,fit['params'],C)
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model = ARIMAForecaster(order,C,cfg['arima_maxiter']).fit(train_y)
                fit = dict(order=order,converged=model.converged,params=model.params.tolist(),warnings=[str(w.message) for w in caught])
                file.write_text(json.dumps(fit,indent=2),encoding='utf-8')
                manifest['arima_fits'].append(fit); save()
            if fit['converged']: arimas[label] = model
            print('ARIMA candidate',label,fit['converged'],flush=True)
        if not arimas: raise RuntimeError('No converged ARIMA candidate')
    for h in cfg['horizons']:
        if done('ARIMA',h,None): continue
        candidates=[]; val_frames={}
        for label,model in arimas.items():
            p=model.forecast_origins(all_y[:offsets['test']], offsets['validation']+indices['validation',h]-1,[h])[h]
            f=attach_distribution(reference('validation',h),censored_normal_distribution(p['mean_raw'],p['std_raw'],C),'censored_normal')
            f['std_raw']=p['std_raw']; val_frames[label]=f
            candidates.append(dict(order=label,val_crps=score_artifact(f,cfg)['crps']))
        best=min(candidates,key=lambda x:x['val_crps']);label=best['order']
        p=arimas[label].forecast_origins(all_y,offsets['test']+indices['test',h]-1,[h])[h]
        f=attach_distribution(reference('test',h),censored_normal_distribution(p['mean_raw'],p['std_raw'],C),'censored_normal');f['std_raw']=p['std_raw']
        record('ARIMA',h,None,{'validation':val_frames[label],'test':f},dict(checkpoint=f'arima_{label}.json',candidates=candidates,best_val_crps=best['val_crps']))

    for seed in cfg['seeds']:
        if not all(done('Prophet',h,seed) for h in cfg['horizons']):
            from src.models.prophet_model import ProphetForecaster
            from prophet.serialize import model_to_json, model_from_json
            file=out/f'prophet_seed{seed}.json'
            model=ProphetForecaster(rated_capacity_mwh=C,**{k:v for k,v in cfg['prophet'].items() if k!='protocol'})
            start=time.perf_counter()
            if file.exists(): model.model=model_from_json(file.read_text())
            else:
                seed_everything(seed); model.fit(raw['train'],seed=seed)
                file.write_text(model_to_json(model.model),encoding='utf-8')
            pred={s:model.predict(raw[s],clip=False) for s in ['validation','test']}
            for h in cfg['horizons']:
                if done('Prophet',h,seed): continue
                frames={}
                for s in pred:
                    f=reference(s,h); f['y_pred_raw']=pred[s][indices[s,h]+h-1]
                    f['y_pred_clipped']=np.clip(f.y_pred_raw,0,C);f['distribution']='dirac';frames[s]=f
                record('Prophet',h,seed,frames,dict(checkpoint=file.name,shared_fit_across_horizons=True,fit_and_prediction_sec=time.perf_counter()-start),'fixed_configuration_no_selection')

        for h in cfg['horizons']:
            for name in NEURAL:
                if done(name,h,seed): continue
                seed_everything(seed); checkpoint=f'{slug(name)}_{h}h_seed{seed}.pt'; start=time.perf_counter()
                print(f'TRAIN seed={seed} +{h}h {name}',flush=True)
                common=dict(lookback_steps=L,horizon=h,batch_size=cfg['batch_size'],device=device,save_model_path=str(out/checkpoint),selection_metric='crps')
                if name==NEURAL[0]:
                    info,pred,_=train_and_evaluate_emfn(arrays['train'],arrays['validation'],arrays['test'],epochs=cfg['emfn_epochs'],patience=cfg['emfn_patience'],lr=cfg['emfn_lr'],capacity_mwh=C,crps_grid_points=cfg['crps_grid_points'],**common)
                else:
                    info,_,pred=train_and_evaluate_torch_model(make_neural(name,L,C),arrays['train'],arrays['validation'],arrays['test'],
                        epochs=cfg['transformer_epochs'] if name==NEURAL[6] else cfg['baseline_epochs'],
                        lr=cfg['cnn_lstm_lr'] if name==NEURAL[4] else cfg['baseline_lr'],rated_capacity_mwh=C,patience=cfg['baseline_patience'],**common)
                frames={s:neural_frame(name,out/checkpoint,arrays[s],raw[s],h,cfg,device) for s in ['validation','test']}
                np.testing.assert_allclose(frames['test'].y_pred_raw,pred,atol=2e-6,rtol=1e-6)
                record(name,h,seed,frames,dict(checkpoint=checkpoint,train_time_sec=time.perf_counter()-start,**{k:info[k] for k in ['best_epoch','stopped_epoch','best_val_loss','best_val_crps','train_count','val_count']}))

            Xtr,ytr,_=flattened_windows(arrays['train'],L,h)
            inputs={s:flattened_windows(arrays[s],L,h) for s in ['validation','test']}
            for name in TREES:
                if done(name,h,seed): continue
                point=name==TREES[0];hurdle=name==TREES[2];start=time.perf_counter()
                model=ProbabilisticLightGBM(cfg['lightgbm'],seed,cfg['torch_threads'],point=point,hurdle=hurdle).fit(Xtr,ytr)
                Xv,yv,_=inputs['validation']; candidates=[]
                for rounds in cfg['lightgbm']['candidate_rounds']:
                    if point:
                        yp,_=model.predict(Xv,rounds);value=float(np.abs(yv-np.clip(yp[:,0],0,C)).mean())
                    else: value=grid_crps(yv,model.distribution(Xv,rounds,C)['cdf'](grid),grid)
                    candidates.append(dict(rounds=rounds,val_crps=value))
                best=min(candidates,key=lambda c:c['val_crps']);rounds=best['rounds']
                directory=f'{slug(name)}_{h}h_seed{seed}_models';model.save(out/directory,rounds)
                frames={}
                for s,(X,y,idx) in inputs.items():
                    f=reference(s,h);yp,p0=model.predict(X,rounds)
                    if point:
                        f['distribution']='dirac';f['y_pred_raw']=yp[:,0];f['y_pred_clipped']=np.clip(yp[:,0],0,C)
                    else:
                        f=attach_distribution(f,quantile_distribution(yp,model.levels,C,p0,cfg['lightgbm']['positive_quantile_floor_mwh']),'hurdle_quantile' if hurdle else 'quantile')
                        for i in range(yp.shape[1]): f[f'raw_q_{i:02d}']=yp[:,i]
                        if hurdle: f['classifier_p_zero']=p0
                    frames[s]=f
                record(name,h,seed,frames,dict(checkpoint_dir=directory,best_val_crps=best['val_crps'],candidates=candidates,rounds=rounds,train_time_sec=time.perf_counter()-start))

            if not done(LONG_TREE,h,seed):
                start=time.perf_counter();features={}
                for s in raw:
                    f,cols=tabular_features(raw[s]);features[s]=f[cols].iloc[indices[s,h]-1]
                labels={s:arrays[s][indices[s,h]+h-1,0] for s in raw}
                params=dict(objective='regression',metric='None',learning_rate=cfg['long_lightgbm']['learning_rate'],num_leaves=cfg['long_lightgbm']['num_leaves'],seed=seed,verbosity=-1,num_threads=cfg['torch_threads'],deterministic=True,force_col_wise=True)
                model=lgb.train(params,lgb.Dataset(features['train'],label=labels['train']),num_boost_round=cfg['long_lightgbm']['rounds'],
                    valid_sets=[lgb.Dataset(features['validation'],label=labels['validation'])],
                    feval=lambda p,d:('crps',float(np.abs(labels['validation']-np.clip(p,0,C)).mean()),False),
                    callbacks=[lgb.early_stopping(cfg['long_lightgbm']['patience'],verbose=False)])
                checkpoint=f'{slug(LONG_TREE)}_{h}h_seed{seed}.txt';model.save_model(str(out/checkpoint));frames={}
                for s in ['validation','test']:
                    f=reference(s,h);f['y_pred_raw']=model.predict(features[s]);f['y_pred_clipped']=np.clip(f.y_pred_raw,0,C);f['distribution']='dirac';frames[s]=f
                record(LONG_TREE,h,seed,frames,dict(checkpoint=checkpoint,best_val_crps=model.best_score['valid_0']['crps'],rounds=model.best_iteration,feature_columns=cols,train_time_sec=time.perf_counter()-start))

    assert len(manifest['experiments'])==234
    assert {e['model'] for e in manifest['experiments']}==set(ALL_MODELS)
    rows=[dict(model=e['model'],seed=e['seed'],horizon_hours=e['horizon_hours'],split=s,selection_basis=e['selection_basis'],**e[s+'_metrics']) for e in manifest['experiments'] for s in ['validation','test']]
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    manifest['status']='complete'
    manifest['artifact_hashes']={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    save()
    # Publication occurs only after independent artifact verification.
    print('COMPLETE',out,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume');args=parser.parse_args()
    with threadpool_limits(limits=4): run(args.resume)
