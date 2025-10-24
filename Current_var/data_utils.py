# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 11:01:28 2025

@author: user
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

COL_VOLT = 3   # 0-based: 4번째 열 → Voltage
COL_R    = 5   #           6번째 열 → Internal resistance
COL_P7   = 6   #           7번째 열 → LGBM target-1
COL_P8   = 7   #           8번째 열 → LGBM target-2
COL_CAP  = 8   #           9번째 열 → 최종 예측 대상

__all__ = ["load_csv", "make_sequences"]

def load_csv(path: Path):
    """CSV 하나를 numpy (float32) 로 가져온다 (skip 4 header rows)."""
    arr = pd.read_csv(path, header=None, skiprows=4).values.astype("float32")
    return arr                             # shape = (T, 9)

def make_sequences(arr, seq_len, hop):
    X = []
    for start in range(0, len(arr) - seq_len, hop):
        X.append(arr[start : start + seq_len])
    return np.asarray(X, np.float32)

