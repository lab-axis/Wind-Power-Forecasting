"""Validation-only checkpoint scores, in MWh on the final forecast support."""
import numpy as np
from scipy.special import betainc
from src.utils.probabilistic_metrics import grid_crps


def validation_crps(y, prediction, capacity=21., grid_points=1001, p_pos=None,
                    alpha=None, beta=None, chunk_size=256):
    y=np.asarray(y,dtype=float).ravel()
    prediction=np.asarray(prediction,dtype=float).ravel()
    if y.shape!=prediction.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Invalid validation targets/predictions')
    if p_pos is None:
        return float(np.mean(np.abs(y-np.clip(prediction,0,capacity))))
    p,a,b=[np.asarray(v,dtype=float).ravel() for v in (p_pos,alpha,beta)]
    if any(v.shape!=y.shape or not np.isfinite(v).all() for v in (p,a,b)) or (p<0).any() or (p>1).any() or (a<=0).any() or (b<=0).any():
        raise ValueError('Invalid hurdle parameters')
    grid=np.linspace(0,capacity,grid_points)
    total=0.
    for i in range(0,len(y),chunk_size):
        s=slice(i,i+chunk_size)
        cdf=1-p[s,None]+p[s,None]*betainc(a[s,None],b[s,None],grid[None,:]/capacity)
        total+=grid_crps(y[s],cdf,grid,return_samples=True).sum()
    return float(total/len(y))


def save_selection_history(checkpoint, history, selection_metric):
    if checkpoint:
        import json
        from pathlib import Path
        Path(str(checkpoint)+'.history.json').write_text(json.dumps(
            dict(selection_metric=selection_metric,epochs=history),indent=2),encoding='utf-8')
