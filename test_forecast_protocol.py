import unittest
import numpy as np
import pandas as pd
import torch
from src.data.forecast_protocol import (align_predictions, valid_window_indices, validate_hourly_frame, prediction_frame)
from src.data.process_saebyeol_weather import clean_hourly_weather
from src.models.emfn import EMFN
from src.models.hurdle_beta import compute_ece, evaluate_zero_head
from src.models.persistence import NaivePersistence, DiurnalPersistence
from src.models.torch_trainer import train_and_evaluate_torch_model
from src.utils.metrics import evaluate_wind_forecast, compute_ramp_metrics

class ForecastProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_batch_invariance_cpu_and_cuda(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(42)
            model = EMFN(lookback_len=24, pred_len=1, n_weather_features=6).to(device).eval()
            target = torch.full((1,24,1), .7, device=device)
            weather = torch.randn(1,24,6,device=device)
            expected = model.predict_point_forecasts(target,weather)
            for n in [1,32,128]:
                x = torch.cat([target,torch.full((n-1,24,1),15.,device=device)])
                w = weather.repeat(n,1,1)
                result = model.predict_point_forecasts(x,w)
                np.testing.assert_allclose(result['y_mean_rmse'][0],expected['y_mean_rmse'][0],atol=1e-4,rtol=1e-5)
                np.testing.assert_allclose(result['p_positive'][0],expected['p_positive'][0],atol=1e-6,rtol=1e-5)

    def test_ece_hand_calculation_and_reversal(self):
        y,p = np.array([0,0,1,1]),np.array([.1,.2,.8,.9])
        self.assertAlmostEqual(compute_ece(y,p,2),.15)
        with self.assertRaises(ValueError): compute_ece(p,y,2)
        self.assertEqual(compute_ece(y,y),0)

    def test_persistence_alignment(self):
        s = pd.Series(np.arange(100))
        for h in [1,3,6,9,12,24]:
            self.assertEqual(DiurnalPersistence().predict(s,h).iloc[50],s.iloc[50+h-24])
            self.assertEqual(NaivePersistence().predict(s,h).iloc[50],s.iloc[50])
        with self.assertRaises(ValueError): DiurnalPersistence().predict(s,25)

    def test_weather_causal_limit_and_flags(self):
        t = pd.date_range('2025-01-01', periods=9, freq='h')
        d = pd.DataFrame({'datetime':t,'aws_wind_speed':[np.nan,2,np.nan,np.nan,np.nan,np.nan,np.nan,9,10]})
        a = clean_hourly_weather(d,3)
        self.assertTrue(np.isnan(a.aws_wind_speed.iloc[0]))
        np.testing.assert_array_equal(a.aws_wind_speed.iloc[2:5], [2,2,2])
        self.assertTrue(a.aws_wind_speed_is_imputed.iloc[4])
        self.assertTrue(np.isnan(a.aws_wind_speed.iloc[5]))
        self.assertEqual(a.aws_wind_speed_age_hours.iloc[5],4)
        d.loc[7:,'aws_wind_speed']=999
        b=clean_hourly_weather(d,3)
        pd.testing.assert_frame_equal(a.iloc[:7],b.iloc[:7])

    def test_future_mutation_preserves_input_and_prediction(self):
        t = pd.date_range('2025-01-01',periods=40,freq='h')
        d = pd.DataFrame({'datetime':t,'aws_wind_speed':np.arange(40,dtype=float)})
        d.loc[20:21,'aws_wind_speed']=np.nan
        a=clean_hourly_weather(d)
        d.loc[24:,'aws_wind_speed']=-999
        b=clean_hourly_weather(d)
        x=torch.ones(1,24,1)
        wa=torch.tensor(a.aws_wind_speed.iloc[:24].to_numpy(),dtype=torch.float32)[None,:,None].repeat(1,1,6)
        wb=torch.tensor(b.aws_wind_speed.iloc[:24].to_numpy(),dtype=torch.float32)[None,:,None].repeat(1,1,6)
        model=EMFN(n_weather_features=6).eval()
        np.testing.assert_array_equal(model.predict_point_forecasts(x,wa)['y_mean_rmse'],model.predict_point_forecasts(x,wb)['y_mean_rmse'])

    def test_missing_rows_restored_without_extrapolation(self):
        d=pd.DataFrame({'datetime':pd.to_datetime(['2025-01-01 00:00','2025-01-01 04:00']),'aws_wind_speed':[2,6]})
        result=clean_hourly_weather(d,2)
        self.assertEqual(len(result),5)
        self.assertTrue(np.isnan(result.aws_wind_speed.iloc[3]))
        validate_hourly_frame(result)
        with self.assertRaises(ValueError): validate_hourly_frame(d)

    def test_window_does_not_compress_missing_time(self):
        d=np.ones((10,2));d[4,1]=np.nan
        self.assertEqual(valid_window_indices(d,3,1),[3,4,8,9])

    def test_raw_violation_and_clipped_point_metrics(self):
        m=evaluate_wind_forecast([0,21],[-1,22])
        self.assertEqual(m['mae'],0)
        self.assertEqual(m['bound_violation_pct'],100)
        self.assertEqual(m['integrated_negative_mwh'],1)

    def test_trainer_preserves_raw_outputs(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.value=torch.nn.Parameter(torch.tensor(-1.))
            def forward(self,x): return self.value.expand(len(x),1)
        data=np.arange(10,dtype=float)
        metrics,yt,yp=train_and_evaluate_torch_model(Constant(),data,data,data,lookback_steps=2,epochs=1,lr=0,device='cpu')
        np.testing.assert_array_equal(yp,np.full(len(yp),-21.))
        self.assertEqual(metrics['bound_violation_pct'],100.)

    def test_common_targets_and_issue_times(self):
        split=pd.DataFrame({'datetime':pd.date_range('2025-01-01',periods=9,freq='h')})
        a=prediction_frame(split,[2,3,4],2,[3,4,5],[3,4,5])
        b=a.iloc[1:].copy()
        aligned=align_predictions({'a':a,'b':b})
        pd.testing.assert_series_equal(aligned['a'].target_time,aligned['b'].target_time)
        b['issue_time']=b.issue_time+pd.Timedelta(hours=1)
        with self.assertRaises(ValueError): align_predictions({'a':a,'b':b})

    def test_duplicate_target_rejected(self):
        f=pd.DataFrame({'target_time':pd.to_datetime(['2025-01-01']*2),'y_true':[1,1],'y_pred_raw':[1,1]})
        with self.assertRaises(ValueError): align_predictions({'a':f})

    def test_ramps_skip_timestamp_gaps(self):
        m=compute_ramp_metrics([0,1,10],[0,1,0],timestamps=pd.to_datetime(['2025-01-01 00:00','2025-01-01 01:00','2025-01-01 03:00']))
        self.assertEqual(m['mae_ramp'],0)

    def test_exact_zero_not_numerical_epsilon(self):
        m=evaluate_zero_head(np.array([0,1e-6,1]),np.array([.01,.99,.99]))
        self.assertEqual(m['f1_zero'],1)

if __name__=='__main__': unittest.main()
