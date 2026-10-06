"""Select presentation rows from verified results; no training or re-scoring."""
import json
import pandas as pd
import yaml
from src.data.forecast_protocol import ROOT
from src.experiments.run_comprehensive_extended_matrix import sha256


def main():
    config_path=ROOT/'configs/paper_presentation.yaml'
    cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    m=json.loads((ROOT/cfg['source_manifest']).read_text(encoding='utf-8'))
    check=json.loads((ROOT/cfg.get('source_validation','reports/multiseed_crps_validation.json')).read_text(encoding='utf-8'))
    provenance=check if 'source_validation' in cfg else json.loads((ROOT/'reports/multiseed_analysis_provenance.json').read_text(encoding='utf-8'))
    if not (m['status']=='complete' and check['status']=='passed' and m['run_id']==check['run_id']==provenance['run_id']):
        raise ValueError('Verified matching source run required')
    source=ROOT/cfg['source_summary']
    if sha256(source)!=provenance['artifact_hashes'][cfg['source_summary']]:
        raise ValueError('Source summary no longer matches verified analysis')
    full=pd.read_csv(source);test=full[full.split.eq('test')]
    columns=['model','horizon_hours','n_fits']+[f'{v}_{suffix}' for v in cfg['metrics'] for suffix in ['mean','sd']]
    outputs={}
    for group,dest in [('main_models','paper_main_standard_baselines.csv'),
                       ('supplementary_probability_models','paper_supplementary_probability.csv')]:
        names=cfg[group]
        if len(names)!=len(set(names)) or 'EMFN (Proposed)' not in names:
            raise ValueError('Unique model roster including EMFN required')
        if group=='main_models' and len(names)-1>cfg['max_comparators_excluding_emfn']:
            raise ValueError('Comparator count exceeds presentation limit')
        view=test[test.model.isin(names)][columns].copy()
        if set(view.model)!=set(names) or len(view)!=len(names)*len(m['config']['horizons']):
            raise ValueError('Missing model/horizon rows in source')
        view['_order']=view.model.map({name:i for i,name in enumerate(names)})
        view=view.sort_values(['horizon_hours','_order']).drop(columns='_order')
        view['run_id']=m['run_id']
        bases={e['model']:e['selection_basis'] for e in m['experiments']}
        view['selection_basis']=view.model.map(bases)
        path=ROOT/'reports/tables'/dest;view.to_csv(path,index=False)
        outputs[path.relative_to(ROOT).as_posix()]=dict(rows=len(view),sha256=sha256(path))
        if group=='main_models' and 'main_table_horizons' in cfg:
            compact=view[view.horizon_hours.isin(cfg['main_table_horizons'])]
            compact_path=ROOT/'reports/tables/paper_main_compact.csv';compact.to_csv(compact_path,index=False)
            outputs[compact_path.relative_to(ROOT).as_posix()]=dict(rows=len(compact),sha256=sha256(compact_path))
    record=dict(run_id=m['run_id'],status='verified_subset_only',config=cfg,
        source_summary_sha256=sha256(source),presentation_config_sha256=sha256(config_path),
        exporter_sha256=sha256(__file__),outputs=outputs)
    (ROOT/'reports/paper_presentation_manifest.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Exported paper views:',{k:v['rows'] for k,v in outputs.items()})


if __name__=='__main__':main()
