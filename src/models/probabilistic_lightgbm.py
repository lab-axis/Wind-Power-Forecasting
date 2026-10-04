"""Matched-input LightGBM point, quantile, and hurdle conditional-quantile models."""
import numpy as np
import lightgbm as lgb
from src.data.forecast_protocol import valid_window_indices
from src.utils.probabilistic_metrics import quantile_distribution


def flattened_windows(array, lookback, horizon):
    indices=np.asarray(valid_window_indices(array,lookback,horizon),dtype=int)
    return (np.stack([array[i-lookback:i].reshape(-1) for i in indices]),
        np.asarray([array[i+horizon-1,0] for i in indices]),indices)


class ProbabilisticLightGBM:
    def __init__(self, config, seed=42, threads=4, hurdle=False, point=False):
        self.config=config;self.hurdle=hurdle;self.point=point
        self.levels=np.asarray(config['quantile_levels'])
        self.params=dict(learning_rate=config['learning_rate'],num_leaves=config['num_leaves'],
            min_data_in_leaf=config['min_data_in_leaf'],verbosity=-1,num_threads=threads,
            seed=seed,deterministic=True,force_col_wise=True)
        self.models=[];self.classifier=None

    def fit(self, X, y):
        rounds=max(self.config['candidate_rounds'])
        if self.hurdle:
            if len(np.unique(y>0))!=2: raise ValueError('Hurdle training requires both zero and positive observations')
            data=lgb.Dataset(X,label=(y>0).astype(int),free_raw_data=False)
            self.classifier=lgb.train(dict(self.params,objective='binary'),data,num_boost_round=rounds)
        mask=(y>0) if self.hurdle else np.ones(len(y),dtype=bool)
        data=lgb.Dataset(X[mask],label=y[mask],free_raw_data=False)
        objectives=[None] if self.point else self.levels
        for level in objectives:
            p=dict(self.params,objective='regression' if self.point else 'quantile')
            if level is not None:p['alpha']=float(level)
            self.models.append(lgb.train(p,data,num_boost_round=rounds))
        return self

    def predict(self, X, rounds):
        raw=np.column_stack([m.predict(X,num_iteration=rounds) for m in self.models])
        p_zero=1-self.classifier.predict(X,num_iteration=rounds) if self.classifier is not None else None
        return raw,p_zero

    def distribution(self, X, rounds, capacity=21.):
        raw,p0=self.predict(X,rounds)
        if self.point:raise ValueError('Point model has no fitted probabilistic distribution')
        return quantile_distribution(raw,self.levels,capacity,p0,self.config['positive_quantile_floor_mwh'])

    def save(self, directory, rounds):
        from pathlib import Path
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for i,m in enumerate(self.models):m.save_model(str(directory/f'quantile_{i:02d}.txt'),num_iteration=rounds)
        if self.classifier is not None:self.classifier.save_model(str(directory/'positive_classifier.txt'),num_iteration=rounds)
