"""Generate colab/wind_power_colab.ipynb (clone -> install -> check -> read stored results).

Kept outside notebooks/ because src.workflows.build_notebooks owns and rewrites that folder.
Run from the repository root: python colab/build_colab_notebook.py
"""
from pathlib import Path
import nbformat as nbf

REPO_URL = "https://github.com/lab-axis/Wind-Power-Forecasting.git"
REPO_DIR = "Wind-Power-Forecasting"

CELLS = [
    ("m", """# Wind Power Forecasting on Google Colab

Clones the repository into the Colab VM, installs the few extra packages, and reads the stored results. **No GPU is needed for this notebook.** A GPU (`Runtime > Change runtime type > T4 GPU`) only helps if you retrain models.

- Stored tables/figures come from `reports/` and `data/`, which are committed to Git.
- `models/runs/` (full training runs and prediction files) is **not** in Git, so plots that need saved predictions or per-epoch history are unavailable until you retrain.
- Retraining everything is long and a free Colab session can disconnect. Start with a single horizon/seed (notebook 04, `RUN_EMFN_EXAMPLE`)."""),
    ("c", f"""import os, pathlib
REPO_URL = "{REPO_URL}"
REPO_DIR = "{REPO_DIR}"
if not pathlib.Path(REPO_DIR).exists():
    !git clone --depth 1 {{REPO_URL}}
%cd {{REPO_DIR}}"""),
    ("c", "!pip -q install -r colab/requirements-colab.txt"),
    ("m", "## Environment check"),
    ("c", """import sys, torch, numpy, pandas
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| pandas", pandas.__version__, "| torch", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
sys.path.insert(0, os.getcwd())
from src.workflows import research as wf
wf.environment()"""),
    ("m", "## Stored results (no training)"),
    ("c", """wf.run_status()
wf.display_scores(models=wf.MAIN, horizons=(1, 6, 24))"""),
    ("m", """## Run the numbered notebooks inside this VM

The notebooks locate the project root from the working directory, so run them in this same VM (opening them separately from GitHub would start a new VM without the clone). Register the kernel name they expect, then execute one, for example notebook 02:

```python
!python -m ipykernel install --user --name wind_power --display-name "Python (wind_power)"
!python -m src.workflows.execute_notebooks --pattern "02_*.ipynb"
```

Executed notebooks are written back into `notebooks/` in this VM (download them from the Files panel). Running with the default `--pattern` rewrites `reports/notebook_execution_validation.json` too, and 03/05 read files under `models/runs/` that are not in Git."""),
]

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
nb.metadata.colab = dict(provenance=[], name="wind_power_colab.ipynb")
nb.cells = [nbf.v4.new_markdown_cell(t) if k == "m" else nbf.v4.new_code_cell(t) for k, t in CELLS]
out = Path(__file__).with_name("wind_power_colab.ipynb")
nbf.write(nb, out)
print("wrote", out)
