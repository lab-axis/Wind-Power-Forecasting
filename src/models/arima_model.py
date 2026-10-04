"""Actual ARIMA(p,d,q), fitted on training data and causally filtered at each origin."""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAForecaster:
    def __init__(self, order=(1,0,1), rated_capacity_mwh=21.0, maxiter=150):
        self.order=tuple(order)
        if len(self.order)!=3 or any(int(x)!=x or x<0 for x in self.order):
            raise ValueError('order must contain nonnegative integer p,d,q')
        self.rated_capacity_mwh=float(rated_capacity_mwh)
        self.maxiter=maxiter
        self.model_res=None
        self.params=None

    @classmethod
    def from_parameters(cls, order, params, rated_capacity_mwh=21.0):
        model=cls(order,rated_capacity_mwh)
        model.params=np.asarray(params,dtype=float)
        if model.params.ndim!=1 or not np.isfinite(model.params).all():
            raise ValueError('Invalid saved ARIMA parameters')
        return model

    def _model(self, values):
        return SARIMAX(np.asarray(values,dtype=float)/self.rated_capacity_mwh,
            order=self.order,trend='c' if self.order[1]==0 else 'n',
            enforce_stationarity=True,enforce_invertibility=True)

    def fit(self, train_series):
        import sys, scipy
        if sys.platform=='win32' and (np.__version__,scipy.__version__)==('2.4.6','1.17.1'):
            raise RuntimeError('Known local Windows state-space binary failure: use the isolated requirements_probabilistic.txt runtime (see reports/probabilistic_baselines.md).')
        values=np.asarray(train_series,dtype=float)
        if values.ndim!=1 or np.isinf(values).any(): raise ValueError('Invalid generation series')
        # NaNs remain on the regular time axis; Kalman filtering handles missing observations.
        self.model_res=self._model(values).fit(disp=False,maxiter=self.maxiter)
        self.params=np.asarray(self.model_res.params)
        self.converged=bool(self.model_res.mle_retvals.get('converged',False))
        return self

    def forecast_origins(self, full_series, origins, horizons):
        """Return raw Gaussian means/stds for y[t+h] using only observations <= t.

        Fitted parameters are fixed. Only FILTERED states are used, never smoothed states.
        The supplied series may contain later observations; they cannot affect an earlier state.
        """
        if self.params is None: raise ValueError('Fit or restore parameters before forecasting')
        origins=np.asarray(origins,dtype=int)
        if not len(origins) or origins.min()<0 or origins.max()>=len(full_series):
            raise ValueError('Invalid origin indices')
        horizons=sorted(set(horizons))
        if not horizons or min(horizons)<1: raise ValueError('Positive horizons required')
        result=self._model(full_series).filter(self.params)
        ssm=result.model.ssm
        T=np.asarray(ssm['transition']); Z=np.asarray(ssm['design'])
        R=np.asarray(ssm['selection']); Q=np.asarray(ssm['state_cov'])
        intercept=np.asarray(ssm['state_intercept'])
        if intercept.ndim==2:
            if not np.allclose(intercept,intercept[:,:1]):
                raise ValueError('Only time-invariant state intercepts are supported')
            intercept=intercept[:,0]
        intercept=intercept.reshape(-1)
        obs_intercept=float(np.asarray(ssm['obs_intercept']).ravel()[0])
        obs_var=float(np.asarray(ssm['obs_cov']).ravel()[0])
        state=result.filtered_state[:,origins].copy()
        covariance=np.moveaxis(result.filtered_state_cov[:,:,origins],2,0).copy()
        noise=R@Q@R.T
        forecasts={}
        for h in range(1,max(horizons)+1):
            state=T@state+intercept[:,None]
            covariance=T[None,:,:]@covariance@T.T[None,:,:]+noise[None,:,:]
            if h in horizons:
                mean=(Z@state).ravel()+obs_intercept
                variance=np.einsum('i,nij,j->n',Z.ravel(),covariance,Z.ravel())+obs_var
                if (variance < -1e-8).any(): raise ValueError('Negative forecast variance')
                forecasts[h]={'mean_raw':mean*self.rated_capacity_mwh,
                    'std_raw':np.sqrt(np.maximum(variance,1e-12))*self.rated_capacity_mwh}
        return forecasts

    def predict_horizon(self, full_series, test_start_idx, test_end_idx, horizon=1):
        # Legacy interface: input window ends before index i; target index i+h-1.
        origins=np.arange(test_start_idx,test_end_idx)-1
        return self.forecast_origins(full_series,origins,[horizon])[horizon]['mean_raw']
