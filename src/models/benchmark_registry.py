"""Single constructor registry shared by training, reloading, and notebooks."""
from src.models.emfn import EMFN
from src.models.dlinear import DLinearForecaster
from src.models.rnn import LSTMForecaster, GRUForecaster
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.tcn_model import TCNForecaster
from src.models.transformer import TimeSeriesTransformerForecaster

NEURAL = ['EMFN (Proposed)', 'DLinear (+Weather)', 'LSTM (+Weather)',
          'GRU (+Weather)', 'CNN-LSTM (+Weather)', 'TCN (+Weather)',
          'Transformer (+Weather)']
TREES = ['LightGBM (24h matched)', 'Quantile LightGBM (24h matched)',
         'Hurdle Quantile LightGBM (24h matched)']
LONG_TREE = 'LightGBM (+Weather)'
FIXED = ['Naive Persistence', 'Diurnal Persistence', 'ARIMA', 'Prophet']
ALL_MODELS = NEURAL + TREES + [LONG_TREE] + FIXED


def make_neural(name, lookback=24, capacity=21.0):
    constructors = {
        NEURAL[0]: lambda: EMFN(lookback_len=lookback, pred_len=1, n_weather_features=6, capacity_mwh=capacity),
        NEURAL[1]: lambda: DLinearForecaster(lookback_steps=lookback, forecast_horizon=1, input_dim=7),
        NEURAL[2]: lambda: LSTMForecaster(input_dim=7, hidden_dim=64, num_layers=2, forecast_horizon=1),
        NEURAL[3]: lambda: GRUForecaster(input_dim=7, hidden_dim=64, num_layers=2, forecast_horizon=1),
        NEURAL[4]: lambda: CNNLSTMForecaster(input_dim=7, lookback_steps=lookback, forecast_horizon=1),
        NEURAL[5]: lambda: TCNForecaster(input_dim=7, num_channels=[64]*3, kernel_size=3, forecast_horizon=1),
        NEURAL[6]: lambda: TimeSeriesTransformerForecaster(input_dim=7, lookback_steps=lookback, forecast_horizon=1),
    }
    return constructors[name]()
