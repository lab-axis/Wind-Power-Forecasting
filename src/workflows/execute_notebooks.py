"""Execute all thin notebooks (training switches remain false); preserve outputs."""
import argparse
import json
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from src.data.forecast_protocol import ROOT
from src.experiments.run_comprehensive_extended_matrix import sha256


def main(pattern='*.ipynb'):
    runtime=ROOT/'.experiment_archive/python_runtime'
    os.environ['PYTHONPATH']=os.pathsep.join([str(runtime),str(ROOT)])
    os.environ['PYTHONUTF8']='1'
    # Keep kernel history/runtime writes inside the project, including in sandboxes.
    os.environ['IPYTHONDIR']=str(ROOT/'.experiment_archive/ipython_notebooks')
    os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.experiment_archive/jupyter_runtime')
    Path(os.environ['IPYTHONDIR']).mkdir(parents=True,exist_ok=True)
    Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(parents=True,exist_ok=True)
    records=[]
    source=ROOT/'reports/all_baselines_manifest.json'
    run_id=json.loads(source.read_text(encoding='utf-8'))['run_id'] if source.exists() else None
    for path in sorted((ROOT/'notebooks').glob(pattern)):
        print('EXECUTE',path.name,flush=True)
        nb=nbformat.read(path,as_version=4)
        client=NotebookClient(nb,timeout=600,kernel_name='wind_power',resources={'metadata':{'path':str(ROOT)}})
        client.execute()
        nb.metadata['research_run_id']=run_id
        nbformat.write(nb,path)
        errors=[o for cell in nb.cells if cell.cell_type=='code' for o in cell.get('outputs',[]) if o.output_type=='error']
        assert not errors
        records.append(dict(file=path.relative_to(ROOT).as_posix(),sha256=sha256(path),
            code_cells=sum(c.cell_type=='code' for c in nb.cells),executed=True))
    out=ROOT/'reports/notebook_execution_validation.json'
    # Partial preflight must not claim the complete five-notebook suite was run.
    if pattern=='*.ipynb':out.write_text(json.dumps(dict(status='passed',run_id=run_id,notebooks=records,training_enabled=False,
        workflow_sha256=sha256(ROOT/'src/workflows/research.py'),template_sha256=sha256(ROOT/'src/workflows/build_notebooks.py')),indent=2),encoding='utf-8')
    print('PASSED',len(records),'notebooks',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pattern',default='*.ipynb');args=parser.parse_args();main(args.pattern)
