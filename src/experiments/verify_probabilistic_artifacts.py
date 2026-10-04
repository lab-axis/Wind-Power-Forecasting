"""Verify extension artifacts, checkpoint reloads and common-sample metrics."""
import json,hashlib,zipfile
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.data.forecast_protocol import ROOT,prepare_splits,align_predictions
from src.models.probabilistic_lightgbm import flattened_windows
from src.models.arima_model import ARIMAForecaster
from src.experiments.run_comprehensive_extended_matrix import sha256,slug
from src.experiments.run_probabilistic_benchmarks import score_artifact,artifact_distribution
from src.utils.probabilistic_metrics import grid_crps


def main():
    m=json.loads((ROOT/'reports/probabilistic_run_manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='complete';cfg=m['config'];run=ROOT/'models/runs'/m['run_id']
    assert sha256(ROOT/'data/processed/merged_dataset.parquet')==m['processed_data_sha256']
    for rel,digest in m['artifact_hashes'].items():assert sha256(run/rel)==digest,rel
    with zipfile.ZipFile(run/'source.zip') as z:
        for rel,digest in m['source_hashes'].items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest,rel
    rows=pd.read_csv(run/'benchmark.csv');candidates=pd.read_csv(run/'validation_candidates.csv')
    raw,arr,_=prepare_splits(pd.read_parquet(ROOT/'data/processed/merged_dataset.parquet'))
    all_y=np.concatenate([raw[k].generation_mwh for k in ['train','validation','test']])
    past=len(raw['train'])+len(raw['validation']);diagnostics=[];refinement=[]
    for h in cfg['horizons']:
        frames={}
        for name in rows[rows.horizon_hours==h].model:
            f=pd.read_parquet(run/f'{slug(name)}_{h}h_test.parquet');frames[name]=f
            assert ((f.target_time-f.issue_time)==pd.Timedelta(hours=h)).all()
            expected=raw['test'].set_index('datetime').loc[f.target_time,'generation_mwh'].to_numpy()
            np.testing.assert_array_equal(f.y_true,expected)
            score=score_artifact(f,cfg);row=rows[(rows.horizon_hours==h)&rows.model.eq(name)].iloc[0]
            for key in ['crps','mae','rmse','count']:
                np.testing.assert_allclose(score[key],row[key],rtol=1e-7,atol=1e-7)
            if f.distribution.iloc[0]!='dirac':
                dist=artifact_distribution(f,cfg)
                np.testing.assert_allclose(dist['mean'],f.y_pred_clipped,rtol=1e-6,atol=1e-6)
                np.testing.assert_allclose(dist['median'],f.y_median,rtol=1e-6,atol=1e-6)
                np.testing.assert_allclose(dist['p_zero'],f.p_zero,rtol=1e-6,atol=1e-6)
                # Numerical sensitivity diagnostic only; does not select models/settings.
                small=f.iloc[::10];ds=artifact_distribution(small,cfg)
                g1=np.linspace(0,cfg['capacity_mwh'],1001);g2=np.linspace(0,cfg['capacity_mwh'],2001)
                c1=grid_crps(small.y_true,ds['cdf'](g1),g1);c2=grid_crps(small.y_true,ds['cdf'](g2),g2)
                refinement.append(dict(model=name,horizon_hours=h,count=len(small),crps_grid1001=c1,crps_grid2001=c2,absolute_difference=abs(c1-c2)))
            qcols=[c for c in f if c.startswith('raw_q_')]
            diagnostics.append(dict(model=name,horizon_hours=h,raw_quantile_violation_pct=100*float(((f[qcols]<0)|(f[qcols]>cfg['capacity_mwh'])).to_numpy().mean()) if qcols else np.nan))
        align_predictions(frames)
        X,_,indices=flattened_windows(arr['test'],cfg['lookback_hours'],h)
        ix=[0,len(indices)//2,len(indices)-1]
        for s in [s for s in m['selections'] if s['horizon_hours']==h]:
            subset=candidates[(candidates.model==s['model'])&(candidates.horizon_hours==h)]
            np.testing.assert_allclose(s['val_crps'],subset.val_crps.min(),atol=1e-12)
            f=frames[s['model']]
            if s['model']=='ARIMA':
                metadata=json.loads((run/s['checkpoint']).read_text(encoding='utf-8'))
                model=ARIMAForecaster.from_parameters(metadata['order'],metadata['params'],cfg['capacity_mwh'])
                result=model.forecast_origins(all_y,past+indices[ix]-1,[h])[h]
                np.testing.assert_allclose(result['mean_raw'],f.y_pred_raw.iloc[ix],atol=1e-7)
                np.testing.assert_allclose(result['std_raw'],f.std_raw.iloc[ix],atol=1e-7)
            else:
                directory=run/s['checkpoint_dir']
                for i,path in enumerate(sorted(directory.glob('quantile_*.txt'))):
                    result=lgb.Booster(model_file=str(path)).predict(X[ix])
                    expected=f.y_pred_raw.iloc[ix] if s['model']=='LightGBM (24h matched)' else f[f'raw_q_{i:02d}'].iloc[ix]
                    np.testing.assert_allclose(result,expected,atol=1e-7)
                if (directory/'positive_classifier.txt').exists():
                    result=lgb.Booster(model_file=str(directory/'positive_classifier.txt')).predict(X[ix])
                    np.testing.assert_allclose(1-result,f.classifier_p_zero.iloc[ix],atol=1e-7)
    pd.DataFrame(diagnostics).to_csv(ROOT/'reports/tables/probabilistic_output_diagnostics.csv',index=False)
    pd.DataFrame(refinement).to_csv(ROOT/'reports/tables/probabilistic_crps_grid_sensitivity.csv',index=False)
    report=dict(status='passed',run_id=m['run_id'],benchmark_rows=len(rows),new_model_horizon_reloads=len(m['selections']),
        validation_selection='all selected values minimize candidate validation CRPS',
        parent_common_samples_preserved=True,raw_and_checkpoint_hashes='verified',
        primary_metrics_recomputed='verified',max_sampled_grid_refinement_difference=max(r['absolute_difference'] for r in refinement))
    (ROOT/'reports/probabilistic_artifact_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
