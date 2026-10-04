"""Comparable CDF-grid CRPS for bounded continuous/atomic forecast distributions."""
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm, beta as beta_dist
from src.utils.metrics import evaluate_wind_forecast, compute_prediction_interval_metrics
from src.models.hurdle_beta import evaluate_zero_head


def grid_crps(y, cdf, grid, return_samples=False):
    y=np.asarray(y,dtype=float).ravel();cdf=np.asarray(cdf,dtype=float);grid=np.asarray(grid,dtype=float)
    if cdf.shape!=(len(y),len(grid)) or np.any(np.diff(grid)<=0): raise ValueError('CDF/grid shape mismatch')
    if not np.isfinite(cdf).all() or (cdf < -1e-8).any() or (cdf > 1+1e-8).any(): raise ValueError('Invalid CDF')
    if (np.diff(cdf,axis=1)<-1e-8).any(): raise ValueError('CDF must be monotone')
    scores=np.trapezoid((cdf-(grid[None,:]>=y[:,None]))**2,grid,axis=1)
    return scores if return_samples else float(scores.mean())


def quantile_distribution(raw_quantiles, levels, capacity=21., p_zero=None, positive_floor=1e-6):
    raw=np.asarray(raw_quantiles,dtype=float);levels=np.asarray(levels,dtype=float)
    if raw.ndim!=2 or raw.shape[1]!=len(levels) or not np.isfinite(raw).all(): raise ValueError('Invalid quantiles')
    if np.any(np.diff(levels)<=0) or levels[0]<=0 or levels[-1]>=1: raise ValueError('Quantile levels must increase within (0,1)')
    conditional=p_zero is not None
    if conditional:
        p_zero=np.asarray(p_zero,dtype=float).ravel()
        if len(p_zero)!=len(raw) or not np.isfinite(p_zero).all() or (p_zero<0).any() or (p_zero>1).any():raise ValueError('Invalid zero probabilities')
    sorted_raw=np.sort(raw,axis=1)
    q=np.clip(sorted_raw,positive_floor if conditional else 0.,capacity)
    values=np.column_stack([np.zeros(len(q)),q,np.full(len(q),capacity)])
    tau=np.r_[0.,levels,1.]
    # Unbounded raw diagnostic: constant tails outside fitted quantile levels.
    raw_values=np.column_stack([sorted_raw[:,0],sorted_raw,sorted_raw[:,-1]])
    raw_mean=np.trapezoid(raw_values,tau,axis=1)
    mean=np.trapezoid(values,tau,axis=1)
    if conditional: mean=mean*(1-p_zero);raw_mean=raw_mean*(1-p_zero)
    else: p_zero=np.array([tau[v==0].max() for v in values])
    def quantile(level):
        if conditional:
            u=np.clip((level-p_zero)/np.maximum(1-p_zero,1e-15),0,1)
            return np.array([np.interp(t,tau,v) if level>p else 0. for t,p,v in zip(u,p_zero,values)])
        return np.array([np.interp(level,tau,v) for v in values])
    def cdf(grid):
        base=np.array([np.interp(grid,v,tau,left=0.,right=1.) for v in values])
        if conditional:
            base=p_zero[:,None]+(1-p_zero[:,None])*base
            base[:,np.asarray(grid)<0]=0.
        return base
    return dict(mean=mean,median=quantile(.5),lower=quantile(.05),upper=quantile(.95),
        raw_mean=raw_mean,p_zero=p_zero,cdf=cdf,
        raw_quantile_violation_pct=100*float(((raw<0)|(raw>capacity)).mean()))


def censored_normal_distribution(mean, std, capacity=21.):
    mu=np.asarray(mean,dtype=float).ravel();sd=np.asarray(std,dtype=float).ravel()
    if mu.shape!=sd.shape or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd<=0).any():raise ValueError('Invalid Gaussian parameters')
    a=-mu/sd;b=(capacity-mu)/sd
    p0=ndtr(a);pC=1-ndtr(b)
    bounded_mean=mu*(ndtr(b)-ndtr(a))+sd*(norm.pdf(a)-norm.pdf(b))+capacity*pC
    def cdf(grid):
        grid=np.asarray(grid);f=ndtr((grid[None,:]-mu[:,None])/sd[:,None])
        f[:,grid<0]=0.;f[:,grid>=capacity]=1.
        return f
    return dict(mean=np.clip(bounded_mean,0,capacity),median=np.clip(mu,0,capacity),
        lower=np.clip(mu+sd*norm.ppf(.05),0,capacity),upper=np.clip(mu+sd*norm.ppf(.95),0,capacity),
        raw_mean=mu,p_zero=p0,cdf=cdf)


def score_distribution(frame, distribution, capacity=21., grid_points=1001):
    d=distribution;y=frame.y_true.to_numpy()
    result=evaluate_wind_forecast(y,d['mean'],y_pred_raw=d['raw_mean'],
        wind_speed=frame.wind_speed_target.to_numpy(),rated_capacity_mwh=capacity,timestamps=frame.target_time)
    result['mae_mean']=result['mae']
    result['mae']=evaluate_wind_forecast(y,d['median'],rated_capacity_mwh=capacity)['mae']
    result['zero_point_f1_mean']=result.pop('zero_f1')
    result['zero_point_balanced_acc_mean']=result.pop('zero_balanced_acc')
    z=evaluate_zero_head(y,1-d['p_zero'])
    result.update(zero_brier=z['brier_score_zero'],zero_auprc=z['auprc_zero'],zero_auroc=z['auroc'],zero_ece=z['ece_zero'])
    result.update(compute_prediction_interval_metrics(y,d['lower'],d['upper'],capacity))
    grid=np.linspace(0,capacity,grid_points)
    result['crps']=grid_crps(y,d['cdf'](grid),grid)
    return result
