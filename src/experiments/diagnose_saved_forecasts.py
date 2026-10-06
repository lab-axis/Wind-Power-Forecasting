"""Descriptive strata and reliability of saved 2024/2025 forecasts; no fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.forecast_protocol import ROOT
from src.experiments.run_comprehensive_extended_matrix import sha256
from src.experiments.run_probabilistic_benchmarks import artifact_distribution
from src.utils.probabilistic_metrics import grid_crps


def observation_losses(frame, cfg):
    y = frame.y_true.to_numpy()
    if frame.distribution.iloc[0] == 'dirac':
        point = np.clip(frame.y_pred_raw.to_numpy(), 0, cfg['capacity_mwh'])
        return pd.DataFrame(dict(crps=np.abs(y-point), absolute_error=np.abs(y-point), squared_error=(y-point)**2))
    parts = []
    grid = np.linspace(0, cfg['capacity_mwh'], cfg['crps_grid_points'])
    for start in range(0, len(frame), 256):
        sub = frame.iloc[start:start+256]
        d = artifact_distribution(sub, cfg)
        truth = sub.y_true.to_numpy()
        lo, hi = d['lower'], d['upper']
        parts.append(pd.DataFrame(dict(crps=grid_crps(truth,d['cdf'](grid),grid,return_samples=True),
            absolute_error=np.abs(truth-d['median']), squared_error=(truth-d['mean'])**2,
            zero_brier=(d['p_zero']-(truth==0))**2, p_zero=d['p_zero'],
            covered_90=((truth>=lo)&(truth<=hi)).astype(float), width_90=hi-lo,
            interval_score_90=hi-lo+20*np.maximum(lo-truth,0)+20*np.maximum(truth-hi,0))))
    return pd.concat(parts, ignore_index=True)


def diagnostic_groups(frame, observations):
    times = pd.DatetimeIndex(frame.target_time)
    seasons = np.array(['winter','winter','spring','spring','spring','summer','summer','summer','autumn','autumn','autumn','winter'])
    previous = observations.generation_mwh.reindex(times-pd.Timedelta(hours=1)).to_numpy()
    truth = frame.y_true.to_numpy()
    transition = np.where(~np.isfinite(previous), 'previous_missing',
        np.where(previous==0,np.where(truth==0,'zero_to_zero','zero_to_positive'),
                            np.where(truth==0,'positive_to_zero','positive_to_positive')))
    issue_wind = observations.aws_wind_speed.reindex(pd.DatetimeIndex(frame.issue_time))
    wind_bin = pd.cut(issue_wind,[-np.inf,3,6,12,np.inf],labels=['below_3','3_to_6','6_to_12','12_or_more'],right=False)
    return dict(all=np.repeat('all',len(frame)), season=seasons[times.month-1],
                target_transition=transition, issue_wind=wind_bin.astype(object).fillna('missing').to_numpy())


def main():
    source = ROOT/'reports/multiseed_crps_manifest.json'
    m = json.loads(source.read_text(encoding='utf-8'))
    assert m['status'] == 'complete'
    run = ROOT/'models/runs'/m['run_id']
    datafile = ROOT/'data/processed/merged_dataset.parquet'
    assert sha256(datafile) == m['processed_data_sha256']
    # Exclude 2026 even from descriptive target-stratum lookups.
    observed = pd.read_parquet(datafile)
    observed = observed[observed.datetime < '2026-01-01'].set_index('datetime')
    rows, reliability, hashes = [], [], {}
    for j,e in enumerate(m['experiments']):
        for split in ['validation','test']:
            p = run/e[split+'_file']
            assert sha256(p) == e[split+'_sha256']
            hashes[p.relative_to(ROOT).as_posix()] = sha256(p)
            frame = pd.read_parquet(p)
            assert frame.target_time.max() < pd.Timestamp('2026-01-01')
            losses = observation_losses(frame,m['config'])
            np.testing.assert_allclose(losses.crps.mean(), e[split+'_metrics']['crps'], atol=1e-9)
            groups = diagnostic_groups(frame,observed)
            meta = dict(model=e['model'],horizon_hours=e['horizon_hours'],seed=e['seed'],split=split)
            for axis,labels in groups.items():
                for label in np.unique(labels):
                    mask = labels == label
                    subset = losses.loc[mask]
                    metrics = {c:float(subset[c].mean()) for c in losses if c not in ('squared_error','absolute_error','p_zero')}
                    rows.append(dict(**meta,stratification=axis,stratum=label,count=int(mask.sum()),
                        zero_prevalence=float((frame.y_true.to_numpy()[mask]==0).mean()),
                        mae=float(subset.absolute_error.mean()),rmse=float(np.sqrt(subset.squared_error.mean())),**metrics))
            if 'p_zero' in losses:
                bins = np.minimum((losses.p_zero.to_numpy()*10).astype(int),9)
                for k in range(10):
                    mask = bins == k
                    reliability.append(dict(**meta,bin=k,lower=k/10,upper=(k+1)/10,count=int(mask.sum()),
                        mean_probability=float(losses.p_zero[mask].mean()) if mask.any() else np.nan,
                        observed_zero_rate=float((frame.y_true.to_numpy()[mask]==0).mean()) if mask.any() else np.nan))
        print(f'DIAGNOSED {j+1}/{len(m["experiments"])} {e["model"]} +{e["horizon_hours"]}h seed={e["seed"]}',flush=True)
    table = pd.DataFrame(rows)
    prefix = ROOT/'reports/tables'
    outputs = [prefix/'forecast_strata_by_seed.csv',prefix/'forecast_strata_summary.csv',prefix/'zero_reliability_by_seed.csv']
    table.to_csv(outputs[0],index=False)
    metric_cols = ['crps','zero_brier','mae','rmse','covered_90','width_90','interval_score_90']
    keys = ['model','horizon_hours','split','stratification','stratum']
    group = table.groupby(keys,dropna=False)
    assert (group['count'].nunique()==1).all()
    summary = group[metric_cols].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary['count'] = group['count'].first()
    summary['zero_prevalence'] = group.zero_prevalence.first()
    summary['n_fits'] = group.size()
    summary.reset_index().to_csv(outputs[1],index=False)
    pd.DataFrame(reliability).to_csv(outputs[2],index=False)
    provenance = dict(status='complete',source_run_id=m['run_id'],source_manifest_sha256=sha256(source),
        script_sha256=sha256(Path(__file__)),processed_data_sha256=sha256(datafile),
        prediction_hashes=hashes,output_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in outputs},
        all_192_crps_scores_reproduced=True,new_training=False,holdout_2026_used=False,
        interpretation='Descriptive analysis after model selection. Validation was used for checkpoint selection; 2025 is a development benchmark. Target-transition strata use realized labels, not available-at-issue predictors. No new tuning or probability recalibration.',
        interval='central 90 percent, inclusive endpoints; point masses can cause coverage above nominal',
        season='DJF/MAM/JJA/SON; validation has July-December only, incomplete summer/winter',
        reliability='fixed ten equal-width bins; empty bins retained; seeds are repeated fits, not independent observations')
    (ROOT/'reports/forecast_diagnostics_manifest.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('COMPLETE',len(table),'stratum rows',flush=True)


if __name__ == '__main__':
    main()
