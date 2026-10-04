"""
PyTorch 기반 시계열 신경망(LSTM, CNN-LSTM, Transformer, DLinear) 통합 학습 및 평가 모듈
- 21.0 MW 설비용량 기반 정규화 [0, 1] 스케일링 적용
- AdamW 최적화 및 Huber Loss를 통한 이상치 강건 학습
- Validation Early Stopping 지원
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Optional
from src.utils.metrics import evaluate_wind_forecast
from src.data.forecast_protocol import valid_window_indices
from src.utils.checkpoint_selection import validation_crps, save_selection_history


class WindTimeSeriesDataset(Dataset):
    """24시간 Lookback -> Horizon h 예측을 위한 시계열 슬라이딩 윈도우 데이터셋 (단변량 및 다변량 기상 지원)"""

    def __init__(
        self,
        series: np.ndarray,
        lookback_steps: int = 24,
        horizon: int = 1,
        scale_factor: float = 21.0,
    ):
        arr = np.array(series, dtype=np.float32).copy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        # 타겟 컬럼(첫 번째 열: 발전량)만 21.0 설비용량으로 스케일링 [0, 1]
        arr[:, 0] = arr[:, 0] / scale_factor
        self.data = arr
        self.lookback = lookback_steps
        self.horizon = horizon

        self.valid_indices = valid_window_indices(self.data, self.lookback, self.horizon)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x = self.data[t - self.lookback : t, :]
        y = self.data[t + self.horizon - 1, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


def train_and_evaluate_torch_model(
    model: nn.Module,
    train_series: np.ndarray,
    val_series: np.ndarray,
    test_series: np.ndarray,
    lookback_steps: int = 24,
    horizon: int = 1,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 0.001,
    rated_capacity_mwh: float = 21.0,
    device: Optional[str] = None,
    save_model_path: Optional[str] = None,
    selection_metric: str = "crps",
    patience: int = 5,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    PyTorch 모델 학습, Validation 기반 Early Stopping, Test 세트 평가 수행
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if selection_metric not in ("crps", "loss"):
        raise ValueError("selection_metric must be crps or loss")
    if patience < 1:
        raise ValueError("patience must be positive")

    model = model.to(device)

    train_ds = WindTimeSeriesDataset(train_series, lookback_steps=lookback_steps, horizon=horizon, scale_factor=rated_capacity_mwh)
    val_ds = WindTimeSeriesDataset(val_series, lookback_steps=lookback_steps, horizon=horizon, scale_factor=rated_capacity_mwh)
    test_ds = WindTimeSeriesDataset(test_series, lookback_steps=lookback_steps, horizon=horizon, scale_factor=rated_capacity_mwh)

    if any(len(ds) == 0 for ds in (train_ds, val_ds, test_ds)):
        raise ValueError("No finite windows in one or more splits")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.HuberLoss(delta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_selection = float("inf")
    history = []
    best_weights = None
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation 평가
        model.eval()
        val_loss = 0.0
        val_predictions = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                p = model(bx)
                val_loss += criterion(p, by).item() * len(by)
                val_predictions.append(p.cpu().numpy().ravel()*rated_capacity_mwh)

        val_loss /= len(val_ds)
        # Use original float64 labels, not scaled float32 round trips.
        val_targets=np.asarray(val_series)[np.asarray(val_ds.valid_indices)+horizon-1]
        if val_targets.ndim==2: val_targets=val_targets[:,0]
        val_crps=validation_crps(val_targets,np.concatenate(val_predictions),rated_capacity_mwh)
        selection=val_crps if selection_metric=="crps" else val_loss
        if not np.isfinite(selection): raise RuntimeError("Nonfinite validation selection score")
        history.append(dict(epoch=epoch+1,val_loss=val_loss,val_crps=val_crps,selection_score=selection))

        if selection < best_selection:
            best_selection = selection
            best_val_crps = val_crps
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        model.train()

    # 최적 가중치 로드
    if best_weights is None:
        raise RuntimeError("No finite validation checkpoint; training failed")
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    if save_model_path:
        from pathlib import Path
        Path(save_model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
        save_selection_history(save_model_path,history,selection_metric)

    # Test 세트 평가
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            p = model(bx).cpu().numpy()
            all_preds.append(p)
            all_trues.append(by.numpy())

    if all_preds:
        y_pred_scaled = np.vstack(all_preds).flatten()
        y_true_scaled = np.vstack(all_trues).flatten()
        # 원래 MWh 스케일로 복원
        y_pred = y_pred_scaled * rated_capacity_mwh
        y_true = y_true_scaled * rated_capacity_mwh
    else:
        y_pred = np.array([])
        y_true = np.array([])

    # Return RAW predictions. The evaluator clips only for point metrics.

    metrics = evaluate_wind_forecast(y_true, y_pred, rated_capacity_mwh=rated_capacity_mwh)
    metrics.update(horizon_hours=horizon, best_epoch=best_epoch, stopped_epoch=epoch + 1,
                   best_val_loss=best_val_loss, train_count=len(train_ds), val_count=len(val_ds))
    metrics.update(selection_metric=selection_metric,best_val_crps=best_val_crps,
                   best_selection_score=best_selection)

    return metrics, y_true, y_pred
