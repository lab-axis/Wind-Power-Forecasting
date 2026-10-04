"""Existing architecture ablations on the corrected protocol (not a pure head study)."""
import argparse
import json
from datetime import datetime, timezone
import pandas as pd
import torch
from src.data.forecast_protocol import ROOT, load_config, prepare_splits, prediction_frame, align_predictions
from src.models.emfn_trainer import train_and_evaluate_emfn
from src.experiments.run_comprehensive_extended_matrix import seed_everything, slug, sha256, add_observations, score_predictions


def main(run_dir):
    run_dir = ROOT / run_dir
    parent = json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    if parent['status'] != 'complete': raise ValueError('Complete the main benchmark first')
    if sha256(ROOT/'data/processed/merged_dataset.parquet') != parent['processed_data_sha256']:
        raise ValueError('Processed data differ from main benchmark')
    config = load_config()
    if config['protocol'] != parent['protocol']: raise ValueError('Protocol differs from main benchmark')
    cfg = config['protocol']
    raw, arrays, scaler = prepare_splits(pd.read_parquet(ROOT/'data/processed/merged_dataset.parquet'),config)
    out = run_dir/'ablation'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(cfg['torch_threads'])
    capacity=config['plant_specs']['max_hourly_mwh']; L=config['modeling']['lookback_hours']
    variants = {
        'EMFN (Endogenous Only)':dict(include_weather=False),
        'EMFN (w/o HF Skips)':dict(use_hf_skips=False),
        'EMFN (w/o Selective Gate)':dict(use_selective_gate=False),
        'EMFN (Deterministic Regression)':dict(regression_mode=True),
    }
    manifest=dict(parent_run_id=parent['run_id'],status='running',protocol=cfg,
        processed_data_sha256=parent['processed_data_sha256'],weather_scaler=scaler,
        interpretation='Architecture plus objective ablations; regression also removes zero route and gate.',
        experiments=[],source_hashes={})
    for p in sorted((ROOT/'src').rglob('*.py')):
        manifest['source_hashes'][p.relative_to(ROOT).as_posix()] = sha256(p)
    import zipfile
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for rel in manifest['source_hashes']: z.write(ROOT/rel,rel)
    rows=[]
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    for h in [1,3]:
        frames={'EMFN (Proposed)':pd.read_parquet(run_dir/f'emfn__proposed_{h}h_common.parquet')}
        for name,options in variants.items():
            seed_everything(cfg['seeds'][0])
            ckpt=out/f'{slug(name)}_{h}h.pt'
            info,_,p=train_and_evaluate_emfn(arrays['train'],arrays['validation'],arrays['test'],
                lookback_steps=L,horizon=h,epochs=cfg['train_epochs_emfn'],batch_size=cfg['batch_size'],
                # Preserve the historical parent model's validation-loss selection.
                capacity_mwh=capacity,device=parent['device'],save_model_path=str(ckpt),selection_metric='loss',**options)
            extra={} if options.get('regression_mode') else dict(y_median=p['y_median'],p_pos=p['p_pos'],alpha=p['alpha'],beta=p['beta'])
            f=prediction_frame(raw['test'],p['input_indices'],h,p['y_true'],p['y_mean'],**extra)
            f=add_observations(f,raw['test'],capacity)
            predfile=out/f'{slug(name)}_{h}h_predictions.parquet'
            f.to_parquet(predfile,index=False);frames[name]=f
            manifest['experiments'].append(dict(model=name,horizon=h,options=options,
                checkpoint=ckpt.name,checkpoint_sha256=sha256(ckpt),prediction_file=predfile.name,
                prediction_sha256=sha256(predfile),training=info))
            save()
            print(f'ABLATION +{h}h {name}: epoch {info["best_epoch"]}, {info["train_time_sec"]:.1f}s',flush=True)
        aligned=align_predictions(frames)
        for name,f in aligned.items():
            f.to_parquet(out/f'{slug(name)}_{h}h_common.parquet',index=False)
            scores=score_predictions(f,capacity,cfg['crps_grid_points'])
            scores.update(model=name,horizon=f'+{h}h',horizon_hours=h,seed=cfg['seeds'][0],
                run_id=parent['run_id'],zero_prevalence=float(f.y_true.eq(0).mean()))
            rows.append(scores)
        pd.DataFrame(rows).to_csv(out/'ablation.csv',index=False)
    manifest.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat())
    save()
    pd.DataFrame(rows).to_csv(ROOT/'reports/tables/emfn_comprehensive_ablation_matrix.csv',index=False)
    (ROOT/'reports/corrected_ablation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ABLATION COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir');main(p.parse_args().run_dir)
