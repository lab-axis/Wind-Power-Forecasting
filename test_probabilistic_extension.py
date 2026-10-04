import unittest
import numpy as np
from src.models.arima_model import ARIMAForecaster
from src.models.probabilistic_lightgbm import flattened_windows,ProbabilisticLightGBM
from src.utils.probabilistic_metrics import quantile_distribution,censored_normal_distribution,grid_crps

class ProbabilisticTests(unittest.TestCase):
    def test_float32_parent_forecast_cdf_stays_within_bounds(self):
        import pandas as pd
        from src.experiments.run_probabilistic_benchmarks import artifact_distribution
        p=np.array([.2,.9,.123456],dtype=np.float32)
        f=pd.DataFrame(dict(distribution=['hurdle_beta']*3,p_pos=p,alpha=[2.]*3,beta=[3.]*3,y_pred_raw=p*8.4,y_median=[1.]*3))
        d=artifact_distribution(f,{'capacity_mwh':21})
        grid=np.linspace(0,21,1001)
        self.assertTrue(np.isfinite(grid_crps([0,1,2],d['cdf'](grid),grid)))

    def test_arima_matches_statsmodels_forecast_with_differencing(self):
        y=np.cumsum(np.random.default_rng(8).normal(size=180))
        m=ARIMAForecaster(order=(1,1,1)).fit(y)
        ours=m.forecast_origins(y,[len(y)-1],[1,3,6])
        expected=m.model_res.get_forecast(6)
        for h in [1,3,6]:
            np.testing.assert_allclose(ours[h]['mean_raw'][0],expected.predicted_mean[h-1]*21,atol=1e-6)
            np.testing.assert_allclose(ours[h]['std_raw'][0],np.sqrt(expected.var_pred_mean[h-1])*21,atol=1e-6)

    def test_arima_future_invariance_and_missing_grid(self):
        y=np.random.default_rng(4).normal(5,1,160)
        m=ARIMAForecaster(order=(1,0,1)).fit(y[:100]);a=y.copy();a[110]=np.nan
        b=a.copy();b[131:]=900
        for h in [1,6]:
            pa=m.forecast_origins(a,[120,130],[h])[h]
            pb=m.forecast_origins(b,[120,130],[h])[h]
            np.testing.assert_allclose(pa['mean_raw'],pb['mean_raw'],atol=1e-10)
            np.testing.assert_allclose(pa['std_raw'],pb['std_raw'],atol=1e-10)

    def test_arima_parameter_reload(self):
        y=np.random.default_rng(9).normal(5,1,150)
        model=ARIMAForecaster(order=(2,0,1)).fit(y)
        restored=ARIMAForecaster.from_parameters(model.order,model.params.tolist())
        a=model.forecast_origins(y,[100,149],[3])[3]
        b=restored.forecast_origins(y,[100,149],[3])[3]
        np.testing.assert_allclose(a['mean_raw'],b['mean_raw'],atol=1e-10)

    def test_gaussian_censoring_mean_and_zero_mass(self):
        d=censored_normal_distribution([0.],[1.],21)
        self.assertAlmostEqual(d['p_zero'][0],.5)
        self.assertAlmostEqual(d['mean'][0],1/np.sqrt(2*np.pi),places=7)
        self.assertEqual(d['median'][0],0)
        np.testing.assert_allclose(d['cdf'](np.array([-1.,0.,21.]))[0],[0,.5,1])

    def test_quantile_distribution_uniform(self):
        levels=np.array([.1,.5,.9]);d=quantile_distribution([[.1,.5,.9]],levels,1.)
        np.testing.assert_allclose(d['mean'],.5)
        self.assertEqual(d['median'][0],.5)
        grid=np.linspace(0,1,10001)
        self.assertAlmostEqual(grid_crps([.5],d['cdf'](grid),grid),1/12,places=7)

    def test_hurdle_distribution_mixture_mean_and_median(self):
        levels=np.array([.1,.5,.9]);d=quantile_distribution([[.1,.5,.9]],levels,1.,p_zero=[.2])
        self.assertAlmostEqual(d['mean'][0],.4)
        self.assertAlmostEqual(d['median'][0],.375)
        np.testing.assert_allclose(d['cdf']([0,.5,1])[0],[.2,.6,1])

    def test_quantile_crossing_and_zero_atoms(self):
        d=quantile_distribution([[2,-1,0]],np.array([.1,.5,.9]),1.)
        self.assertEqual(d['p_zero'][0],.5)
        f=d['cdf'](np.linspace(0,1,100))
        self.assertTrue((np.diff(f)>=0).all())
        self.assertTrue(0<=d['mean'][0]<=1)
        self.assertGreater(d['raw_quantile_violation_pct'],0)

    def test_flattened_windows_target_alignment(self):
        x=np.column_stack([np.arange(12.),np.arange(12.)+100])
        X,y,idx=flattened_windows(x,3,2)
        np.testing.assert_array_equal(X[0],x[:3].reshape(-1))
        self.assertEqual(y[0],4);self.assertEqual(idx[0],3)

    def test_lightgbm_probability_and_round_serialization(self):
        import tempfile
        from pathlib import Path
        import lightgbm as lgb
        rng=np.random.default_rng(2);X=rng.normal(size=(180,4));y=np.maximum(0,X[:,0]+1)
        cfg=dict(quantile_levels=[.1,.5,.9],candidate_rounds=[5,10],num_leaves=5,learning_rate=.1,min_data_in_leaf=5,positive_quantile_floor_mwh=1e-6)
        m=ProbabilisticLightGBM(cfg,threads=2,hurdle=True).fit(X,y)
        raw,p=m.predict(X[:5],5);d=m.distribution(X[:5],5)
        self.assertTrue(((p>=0)&(p<=1)).all());self.assertTrue((d['mean']>=0).all())
        with tempfile.TemporaryDirectory() as td:
            m.save(td,5);reloaded=lgb.Booster(model_file=str(Path(td)/'quantile_00.txt'))
            np.testing.assert_allclose(reloaded.predict(X[:5]),raw[:,0])

if __name__=='__main__':unittest.main()
