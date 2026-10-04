"""Verify saved data/prediction/checkpoint identities without retraining."""
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import torch
from src.data.forecast_protocol import ROOT, load_config, prepare_splits, align_predictions
from src.models.emfn import EMFN
from src.models.emfn_trainer import EMFNMultiChannelDataset
from src.experiments.run_comprehensive_extended_matrix import sha256, slug
from src.utils.metrics import evaluate_wind_forecast


def main():
    m=json.loads((ROOT/'reports/corrected_run_manifest.json').read_text(encoding='utf-8'))
    run=ROOT/'models/runs'/m['run_id']
    assert m['status']=='complete'
    assert sha256(ROOT/'data/processed/merged_dataset.parquet')==m['processed_data_sha256']
    for rel,digest in m['raw_hashes'].items(): assert sha256(ROOT/rel)==digest,rel
    with zipfile.ZipFile(run/'source.zip') as z:
        for rel,digest in m['source_hashes'].items(): assert hashlib.sha256(z.read(rel)).hexdigest()==digest,rel
    frame=pd.read_parquet(ROOT/'data/processed/merged_dataset.parquet')
    config=load_config(); raw,arr,scaler=prepare_splits(frame,config)
    assert scaler==m['weather_scaler']
    bench=pd.read_csv(run/'benchmark.csv')
    for entry in m['experiments']:
        assert sha256(run/entry['prediction_file'])==entry['prediction_sha256']
        if entry['checkpoint']: assert sha256(run/entry['checkpoint'])==entry['checkpoint_sha256']
    checked=0
    diagnostics=[]
    torch.set_num_threads(4)
    for h in m['horizons']:
        frames={}
        for entry in [e for e in m['experiments'] if e['horizon']==h]:
            name=entry['model'];p=pd.read_parquet(run/f'{slug(name)}_{h}h_common.parquet')
            frames[name]=p
            assert ((p.target_time-p.issue_time)==pd.Timedelta(hours=h)).all()
            np.testing.assert_array_equal(p.y_pred_clipped,np.clip(p.y_pred_raw,0,21))
            expected=raw['test'].set_index('datetime').loc[p.target_time,'generation_mwh'].to_numpy()
            np.testing.assert_array_equal(p.y_true,expected)
            scores=evaluate_wind_forecast(p.y_true,p.y_pred_raw,timestamps=p.target_time)
            row=bench[(bench.horizon_hours==h)&bench.model.eq(name)].iloc[0]
            for metric in ['count','mae','rmse','bound_violation_pct','min_pred_mwh','max_pred_mwh']:
                np.testing.assert_allclose(scores[metric],row[metric],rtol=1e-8,atol=1e-8)
            checked+=1
            diagnostics.append(dict(model=name,horizon_hours=h,count=len(p),
                raw_std=float(p.y_pred_raw.std()),clipped_std=float(p.y_pred_clipped.std()),
                clipped_zero_pct=100*float(p.y_pred_clipped.eq(0).mean()),
                constant_prediction=bool(p.y_pred_clipped.nunique()==1)))
        align_predictions(frames)
        ds=EMFNMultiChannelDataset(arr['test'],24,h)
        model=EMFN().eval()
        model.load_state_dict(torch.load(run/f'emfn_{h}h.pt',map_location='cpu',weights_only=True))
        indices=[0,len(ds)//2,len(ds)-1]
        x=torch.stack([ds[i][0] for i in indices]);w=torch.stack([ds[i][1] for i in indices])
        prediction=model.predict_point_forecasts(x,w)
        recorded=pd.read_parquet(run/f'emfn__proposed_{h}h_predictions.parquet')
        np.testing.assert_allclose(prediction['y_mean_rmse'].ravel(),recorded.y_pred_raw.iloc[indices],atol=2e-4,rtol=1e-4)
        metadata=json.loads((ROOT/f'models/checkpoints/emfn_{h}h.metadata.json').read_text(encoding='utf-8'))
        assert sha256(ROOT/f'models/checkpoints/emfn_{h}h.pt')==metadata['checkpoint_sha256']
    am=json.loads((ROOT/'reports/corrected_ablation_manifest.json').read_text(encoding='utf-8'))
    assert am['parent_run_id']==m['run_id'] and am['status']=='complete'
    for entry in am['experiments']:
        for key in ['checkpoint','prediction_file']:
            hashkey='checkpoint_sha256' if key=='checkpoint' else 'prediction_sha256'
            assert sha256(run/'ablation'/entry[key])==entry[hashkey]
    report={'run_id':m['run_id'],'status':'passed','benchmark_rows_verified':checked,
        'ablation_artifact_pairs_verified':len(am['experiments']),'emfn_checkpoints_reloaded':len(m['horizons']),
        'checks':['raw data hashes','processed data hash','source snapshot hashes','prediction/checkpoint hashes',
        'identical targets and issue times','raw/clipped arrays','recomputed point and bound metrics',
        'CPU checkpoint reload agrees with CUDA recorded predictions within tolerance']}
    (ROOT/'reports/artifact_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame(diagnostics).to_csv(ROOT/'reports/tables/prediction_diagnostics.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
