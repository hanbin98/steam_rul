# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 13:46:02 2025

@author: user
"""

import numpy as np, tensorflow as tf, joblib
from pathlib import Path
from tensorflow.keras.layers import (Input, LSTM, Conv1D, concatenate,
                                     Flatten, Dense, LeakyReLU, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler

from data_utils import load_csv, make_sequences, COL_VOLT, COL_R, COL_P7, COL_P8, COL_CAP

SEQ   = 100     # sequence length
HOP   = 1       # stride
LR    = 1e-4
BATCH = 64
EPOCH = 200

DATA_DIR = Path("data")
LGBM_DIR = Path("models/lgbm")

model_p7 = lgb.Booster(model_file=str(LGBM_DIR / "lgbm_p7.txt"))
model_p8 = lgb.Booster(model_file=str(LGBM_DIR / "lgbm_p8.txt"))
model_v  = lgb.Booster(model_file=str(LGBM_DIR / "lgbm_v.txt"))
model_ir = lgb.Booster(model_file=str(LGBM_DIR / "lgbm_ir.txt"))

def fill_and_stack(path):
    arr = load_csv(path)
    feats = arr[:, [0, 1, 2, 4]].astype(float)
    feats[:, 1] /= 100.

    arr[:, COL_P7]   = model_p7.predict(feats)
    arr[:, COL_P8]   = model_p8.predict(feats)
    arr[:, COL_VOLT] = model_v.predict(feats)
    arr[:, COL_R]    = model_ir.predict(feats)

    ch_v  = arr[:, COL_VOLT:COL_VOLT+1]
    ch_r  = arr[:, COL_R   :COL_R+1]
    ch_p7 = arr[:, COL_P7  :COL_P7+1]
    ch_p8 = arr[:, COL_P8  :COL_P8+1]

    cap_raw = arr[:, COL_CAP:COL_CAP+1]
    
    cap_scaled = MinMaxScaler().fit_transform(cap_raw)
    idx_start  = np.arange(0, len(cap_scaled) - SEQ, HOP)
    y          = cap_scaled[idx_start + SEQ, 0]
    
    v_seq  = make_sequences(ch_v ,  SEQ, HOP)
    r_seq  = make_sequences(ch_r ,  SEQ, HOP)
    p7_seq = make_sequences(ch_p7, SEQ, HOP)
    p8_seq = make_sequences(ch_p8, SEQ, HOP)
    
    return v_seq, r_seq, p7_seq, p8_seq, y


train_files = sorted((DATA_DIR/"train").glob("*.csv"))
test_files  = sorted((DATA_DIR/"test").glob("*.csv"))

def stack_files(files):
    Vs, Rs, P7s, P8s, Ys = [],[],[],[],[]
    for p in files:
        v,r,p7,p8,y = fill_and_stack(p)
        Vs.append(v); Rs.append(r); P7s.append(p7); P8s.append(p8); Ys.append(y)
    return map(lambda lst: np.concatenate(lst,0), (Vs, Rs, P7s, P8s, Ys))

trV,trR,trP7,trP8,trY = stack_files(train_files)
ttV,ttR,ttP7,ttP8,ttY = stack_files(test_files)


inp_v  = Input((SEQ,1));  v_feat  = Conv1D(32,5,padding='same',activation='relu')(inp_v)
inp_r  = Input((SEQ,1));  r_feat  = Conv1D(32,5,padding='same',activation='relu')(inp_r)
inp_p7 = Input((SEQ,1));  p7_feat = Conv1D(32,5,padding='same',activation='relu')(inp_p7)
inp_p8 = Input((SEQ,1));  p8_feat = Conv1D(32,5,padding='same',activation='relu')(inp_p8)

cnn_all = concatenate([v_feat,r_feat,p7_feat,p8_feat])
cnn_all = Conv1D(32,5,padding='same',activation='relu')(cnn_all)

lstm_in = concatenate([inp_v, inp_r, inp_p7, inp_p8])
lstm_feat = LSTM(32, return_sequences=True)(lstm_in)

merged = concatenate([cnn_all, lstm_feat])
x = Flatten()(merged)
x = Dense(64)(x); x = LeakyReLU(0.05)(x); x = Dropout(0.3)(x)
out = Dense(1)(x)

model = Model([inp_v,inp_r,inp_p7,inp_p8], out)
model.compile(loss='mse', optimizer=Adam(LR))
model.summary()


model.fit([trV, trR, trP7, trP8],
          trY,
          validation_split=0.1,
          epochs=EPOCH,
          batch_size=BATCH,
          shuffle=True)


#%% Variables save

import json
from pathlib import Path

EXPORT_DIR = Path("export")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
NPZ_PATH = EXPORT_DIR / "cnn_lstm_portable.npz"


weights_list = model.get_weights()
n_weights = len(weights_list)

meta = {
    "SEQ": SEQ,
    "HOP": HOP,
    "LR": LR,
    "BATCH": BATCH,
    "EPOCH": EPOCH,
    "inputs": ["V", "R", "P7", "P8"],
    "target": "Capacity(next step; per-window label)",
    "framework": "tensorflow-keras",
    "keras_version": tf.__version__,
    "notes": "LightGBM 보정 특성(V, R, P7, P8)을 윈도우(SEQ)로 입력하여 다음 시점 용량을 예측.",
}

try:
    meta.update({
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_mape": float(mape),
        "test_r2": float(r2),
    })
except NameError:
    pass


arch_json = model.to_json()


to_save = {f"w_{i:03d}": w for i, w in enumerate(weights_list)}
to_save["n_weights"] = np.array(n_weights, dtype=np.int32)
to_save["architecture_json"] = np.frombuffer(arch_json.encode("utf-8"), dtype=np.uint8)
to_save["meta_json"] = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"),
                                     dtype=np.uint8)

np.savez_compressed(NPZ_PATH, **to_save)
print(f"✔ Exported portable NPZ → {NPZ_PATH}")












