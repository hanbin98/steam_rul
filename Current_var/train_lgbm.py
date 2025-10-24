# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:07:55 2025

@author: user
"""

import joblib, json, re, os, random, numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from data_utils import load_csv, COL_P7, COL_P8

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

DATA_DIR = Path("data")           # ← 경로만 수정하세요
SAVE_DIR = Path("models/lgbm"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────── CSV 모두 묶어 하나의 테이블 생성 ─────────────────
train_files = sorted((DATA_DIR/"train").glob("*.csv"),
                     key=lambda p: int(re.sub(r"\D", "", p.stem)))
full = np.concatenate([load_csv(p) for p in train_files], axis=0)   # (ΣT, 9)

X = full[:, [0,1,2,4]].copy()
X[:,1] /= 100.                 # SOC (2번째 열) /100
y7 = full[:, COL_P7]
y8 = full[:, COL_P8]
vol = full[:, 3]
ir = full[:, 5]

def train_one(y, save_path):
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=.2, random_state=SEED)
    lgb_train = lgb.Dataset(Xtr, ytr)
    lgb_valid = lgb.Dataset(Xva, yva)

    params = dict(objective="regression", metric="rmse",
                  learning_rate=0.0431776, num_leaves=192, max_depth=4,
                  min_child_samples=35,
                  subsample=0.6274, subsample_freq=1,
                  colsample_bytree=0.9314,
                  lambda_l1=0.558, lambda_l2=1.557,
                  seed=SEED, verbose=-1)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=50000,
                    valid_sets=[lgb_valid],
                    callbacks=[lgb.early_stopping(500, first_metric_only=True)])

    rmse = mean_squared_error(yva,
                              gbm.predict(Xva, num_iteration=gbm.best_iteration),
                              squared=False)
    mape = mean_absolute_percentage_error(yva, gbm.predict(Xva, num_iteration=gbm.best_iteration))
    r2 = r2_score(yva, gbm.predict(Xva, num_iteration=gbm.best_iteration))
    print(f"RMSE = {rmse:.4f}  ‖  best_iter = {gbm.best_iteration}")
    print(f"MAPE = {mape:.4f}")
    print(f"R2 = {r2:.4f}")

    gbm.save_model(save_path)
    return gbm

mdl_p7 = train_one(y7, SAVE_DIR/"lgbm_p7.txt")
mdl_p8 = train_one(y8, SAVE_DIR/"lgbm_p8.txt")
mdl_v = train_one(vol, SAVE_DIR/"lgbm_v.txt")
mdl_ir = train_one(ir, SAVE_DIR/"lgbm_ir.txt")



print("\n>>> LGBM 두 모델 저장 완료")
