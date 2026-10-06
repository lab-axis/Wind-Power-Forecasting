"""Prepare controls and smoke-test training without opening 2026 or scoring 2025.

Default: write a readiness manifest. --smoke: synthetic one-epoch checks.
--run: explicitly execute the prepared validation-only experiment (126 fits).
"""
import argparse
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import torch
import yaml
from src.data.forecast_protocol import ROOT,prepare_splits,prediction_frame,valid_window_indices
from src.models.emfn import EMFN
from src.models.emfn_ablation import SharedAdapterEMFN,CommonRepresentationEMFN
from src.models.emfn_trainer import train_and_evaluate_emfn
from src.experiments.run_comprehensive_extended_matrix import seed_everything,sha256,add_observations
from src.experiments.run_probabilistic_benchmarks import score_artifact


def variant_options(name):
    if name=='reference': return {}
    if name=='shared_adapter': return dict(model_factory=SharedAdapterEMFN)
    if name=='no_weather_state_features': return dict(use_selective_gate=False)
    if name=='no_hf_ar': return dict(use_hf_skips=False)
    if name.startswith('common_'):
        head = name.removeprefix('common_')
        return dict(model_factory=partial(CommonRepresentationEMFN,head_kind=head),regression_mode=head!='hurdle')
    raise ValueError(name)


def main(smoke=False, run=False):
    cfgfile = ROOT/'configs/structural_ablation.yaml'
    cfg = yaml.safe_load(cfgfile.read_text(encoding='utf-8'))
    source = ROOT/'data/processed/merged_dataset.parquet'
    m = json.loads((ROOT/'reports/multiseed_crps_manifest.json').read_text(encoding='utf-8'))
    assert sha256(source)==m['processed_data_sha256']
    torch.set_num_threads(cfg['torch_threads'])
    counts = {}
    for variant in cfg['variants']:
        seed_everything(cfg['seeds'][0])
        opts = variant_options(variant);factory = opts.pop('model_factory',EMFN)
        model = factory(lookback_len=cfg['lookback_hours'],n_weather_features=6,capacity_mwh=cfg['capacity_mwh'],**opts)
        counts[variant] = sum(p.numel() for p in model.parameters())
    record = dict(status='prepared',config=cfg,config_sha256=sha256(cfgfile),parameter_counts=counts,
        processed_data_sha256=sha256(source),planned_fits=len(cfg['variants'])*len(cfg['horizons'])*len(cfg['seeds']),
        real_data_training_performed=False,holdout_2026_used=False)
    files = [Path(__file__),ROOT/'src/models/emfn.py',ROOT/'src/models/emfn_ablation.py',ROOT/'src/models/emfn_trainer.py']
    record['source_hashes'] = {p.relative_to(ROOT).as_posix():sha256(p) for p in files}
    data = pd.read_parquet(source)
    data = data[data.datetime<'2025-01-01']
    raw,arrays,scaler = prepare_splits(data)
    record['available_windows'] = {split:{str(h):len(valid_window_indices(arrays[split],cfg['lookback_hours'],h))
        for h in cfg['horizons']} for split in ('train','validation')}
    assert all(n>0 for counts in record['available_windows'].values() for n in counts.values())
    if smoke:
        rng = np.random.default_rng(1701)
        a = rng.normal(size=(80,7));a[:,0]=rng.uniform(0,21,80);a[::5,0]=0
        smoke_rows = []
        for name in cfg['variants']:
            seed_everything(42)
            info,_,p = train_and_evaluate_emfn(a,a,a,lookback_steps=24,horizon=1,epochs=1,
                d_model=8,dilations=(1,),batch_size=32,device='cpu',crps_grid_points=101,
                model_name=name,**variant_options(name))
            assert np.isfinite(p['y_mean']).all() and info['best_epoch']==1
            smoke_rows.append(dict(variant=name,best_val_crps=info['best_val_crps'],count=len(p['y_true'])))
        record['synthetic_smoke'] = smoke_rows
        record['status'] = 'prepared_and_smoke_passed'
    (ROOT/'reports/structural_ablation_readiness.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
    if not run: return
    # Explicit bounded scope: no 2025/2026 labels are passed to the trainer.
    out = ROOT/'models/runs'/('structural_validation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    manifest = dict(record,status='running',run_id=out.name,weather_scaler=scaler,experiments=[],source_hashes={})
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').glob('*.yaml')):
            rel=p.relative_to(ROOT).as_posix();manifest['source_hashes'][rel]=sha256(p);z.write(p,rel)
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save()
    for seed in cfg['seeds']:
        for h in cfg['horizons']:
            for name in cfg['variants']:
                seed_everything(seed);checkpoint=out/f'{name}_{h}h_seed{seed}.pt'
                info,_,p=train_and_evaluate_emfn(arrays['train'],arrays['validation'],arrays['validation'],
                    lookback_steps=cfg['lookback_hours'],horizon=h,epochs=cfg['epochs'],patience=cfg['patience'],
                    lr=cfg['learning_rate'],batch_size=cfg['batch_size'],capacity_mwh=cfg['capacity_mwh'],
                    crps_grid_points=cfg['crps_grid_points'],device='cuda' if torch.cuda.is_available() else 'cpu',
                    save_model_path=str(checkpoint),model_name=name,**variant_options(name))
                extra={} if info['regression_mode'] else {k:p[k] for k in ('y_median','p_pos','p_zero','alpha','beta')}
                f=prediction_frame(raw['validation'],p['input_indices'],h,p['y_true'],p['y_mean'],**extra)
                f=add_observations(f,raw['validation'],cfg['capacity_mwh'])
                f['distribution']='dirac' if info['regression_mode'] else 'hurdle_beta'
                metrics=score_artifact(f,cfg)
                np.testing.assert_allclose(metrics['crps'],info['best_val_crps'],atol=2e-6,rtol=1e-6)
                pred=checkpoint.with_suffix('.parquet');f.to_parquet(pred,index=False)
                manifest['experiments'].append(dict(variant=name,seed=seed,horizon=h,training=info,metrics=metrics,
                    checkpoint=checkpoint.name,checkpoint_sha256=sha256(checkpoint),prediction_file=pred.name,prediction_sha256=sha256(pred)))
                save();print('DONE',len(manifest['experiments']),record['planned_fits'],flush=True)
    manifest.update(status='complete',real_data_training_performed=True);save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--run',action='store_true')
    args=p.parse_args();main(args.smoke,args.run)
