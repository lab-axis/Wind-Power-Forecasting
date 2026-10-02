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

        self.valid_indices = []
        max_idx = len(self.data) - self.horizon
        for i in range(self.lookback, max_idx + 1):
            if not np.isnan(self.data[i + self.horizon - 1, 0]):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x = self.data[t - self.lookback : t, :]
        x = np.nan_to_num(x, nan=0.0)
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
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    PyTorch 모델 학습, Validation 기반 Early Stopping, Test 세트 평가 수행
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    train_ds = WindTimeSeriesDataset(train_series, lookback_steps=lookback_steps, horizon=horizon, scale_factor=rated_capacity_mwh)
    val_ds = WindTimeSeriesDataset(val_series, lookback_steps=lookback_steps, horizon=horizon, scale_factor=rated_capacity_mwh)
    test_ds = WindTimeSeriesDataset(test_series, lookback_steps=lookback_steps, horizon=horizon, scale_factor=rated_capacity_mwh)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.HuberLoss(delta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_weights = None
    patience = 5
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
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                p = model(bx)
                val_loss += criterion(p, by).item()

        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        model.train()

    # 최적 가중치 로드
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

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

    # 물리적 상한 클리핑
    y_pred = np.clip(y_pred, 0.0, rated_capacity_mwh)

    metrics = evaluate_wind_forecast(y_true, y_pred, rated_capacity_mwh=rated_capacity_mwh)
    metrics["horizon_hours"] = horizon

    return metrics, y_true, y_pred
