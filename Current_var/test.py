# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:36:26 2025

@author: user
"""

import os
import json
import numpy as np
import tensorflow as tf
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# 학습 때 사용한 유틸 불러오기 (컬럼 인덱스/시퀀스 생성/CSV 로더)
from data_utils import load_csv, make_sequences, COL_VOLT, COL_R, COL_P7, COL_P8, COL_CAP

# ---------------------------------------------------------------------
# 1) 저장된 Keras 모델(아키텍처+가중치) 로드
# ---------------------------------------------------------------------
NPZ_PATH = Path("export/cnn_lstm_portable.npz")
data = np.load(NPZ_PATH, allow_pickle=False)

arch_json = bytes(data["architecture_json"]).decode("utf-8")
meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
n = int(data["n_weights"])
weights = [data[f"w_{i:03d}"] for i in range(n)]

model = tf.keras.models.model_from_json(arch_json)
model.set_weights(weights)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(meta["LR"]))

# 메타에서 SEQ/HOP을 가져와 학습 설정과 동일하게 사용
SEQ = int(meta.get("SEQ", 100))
HOP = int(meta.get("HOP", 1))

# ---------------------------------------------------------------------
# 2) LightGBM 보정 모델 로드 (학습 때와 동일 경로/파일명 가정)
# ---------------------------------------------------------------------
LGBM_DIR = Path("models/lgbm")
model_p7 = lgb.Booster(model_file=str(LGBM_DIR / "lgbm_p7.txt"))
model_p8 = lgb.Booster(model_file=str(LGBM_DIR / "lgbm_p8.txt"))
model_v  = lgb.Booster(model_file=str(LGBM_DIR / "lgbm_v.txt"))
model_ir = lgb.Booster(model_file=str(LGBM_DIR / "lgbm_ir.txt"))

# ---------------------------------------------------------------------
# 3) 시퀀스 생성 함수 (학습 때와 동일)
# ---------------------------------------------------------------------
def fill_and_stack(path: Path):
    """한 개 CSV에서 (V,R,P7,P8) 시퀀스와 스케일된 용량 라벨 y를 생성"""
    arr = load_csv(path)                       # (T, 9)
    feats = arr[:, [0, 1, 2, 4]].astype(float) # (time, SOC%, temp, C-rate 등 가정)
    feats[:, 1] /= 100.                        # SOC(% → 0~1)

    # LightGBM 보정 예측치로 채우기 (원-스케일 그대로)
    arr[:, COL_P7]   = model_p7.predict(feats)
    arr[:, COL_P8]   = model_p8.predict(feats)
    arr[:, COL_VOLT] = model_v.predict(feats)
    arr[:, COL_R]    = model_ir.predict(feats)

    # 채널 선택
    ch_v  = arr[:, COL_VOLT:COL_VOLT+1]  # (T,1)
    ch_r  = arr[:, COL_R   :COL_R+1]
    ch_p7 = arr[:, COL_P7  :COL_P7+1]
    ch_p8 = arr[:, COL_P8  :COL_P8+1]

    # 용량 스케일러 (파일별 min-max)
    cap_raw = arr[:, COL_CAP:COL_CAP+1]
    cap_scaled = MinMaxScaler().fit_transform(cap_raw)           # (T,1)

    # 시퀀스 인덱스 구성
    idx_start = np.arange(0, len(cap_scaled) - SEQ, HOP)
    if len(idx_start) == 0:
        # 길이가 부족하면 빈 배열 반환
        return (np.empty((0, SEQ, 1)),)*4 + (np.empty((0,), dtype=float),)

    # 라벨: 윈도우 다음 시점의 스케일된 용량
    y = cap_scaled[idx_start + SEQ, 0]                           # (N,)

    # 입력 시퀀스
    v_seq  = make_sequences(ch_v ,  SEQ, HOP)                    # (N,SEQ,1)
    r_seq  = make_sequences(ch_r ,  SEQ, HOP)
    p7_seq = make_sequences(ch_p7, SEQ, HOP)
    p8_seq = make_sequences(ch_p8, SEQ, HOP)

    return v_seq, r_seq, p7_seq, p8_seq, y

# ---------------------------------------------------------------------
# 4) 테스트 파일에서 53/57/59번만 선택해 예측/플롯 저장
# ---------------------------------------------------------------------
DATA_DIR = Path("data")
TEST_DIR = DATA_DIR / "test"

target_names = ["battery_53_info.csv", "battery_57_info.csv", "battery_59_info.csv"]

# 존재하는 것만 수집 + 누락 파일 경고
selected_test_files = []
missing = []
for name in target_names:
    p = TEST_DIR / name
    if p.exists():
        selected_test_files.append(p)
    else:
        missing.append(name)

print("[INFO] Selected test files (by filename):", [p.name for p in selected_test_files])
if missing:
    print("[WARN] Missing files in", TEST_DIR, "→", missing)

SAVE_PLOT_DIR = Path("plots")
SAVE_PLOT_DIR.mkdir(exist_ok=True, parents=True)

def smooth(arr, win=11):
    """Trimmed-mean smoothing (양 끝값 1개 제외)"""
    if win < 3: return np.asarray(arr, dtype=float)
    hw = win // 2
    out = []
    for i in range(len(arr)):
        w = arr[max(0, i-hw): min(len(arr), i+hw+1)]
        if len(w) > 2:
            sw = np.sort(w)
            out.append(np.mean(sw[1:-1]))
        else:
            out.append(np.mean(w))
    return np.asarray(out, dtype=float)

for csv_path in selected_test_files:
    # --- 입력 시퀀스 생성 ---
    v, r, p7, p8, y_scaled = fill_and_stack(csv_path)
    if len(v) == 0:
        print(f"[skip] {csv_path.name}  (길이가 SEQ={SEQ} 미만)")
        continue

    # --- 스케일된 출력 예측 ---
    pred_scaled = model.predict([v, r, p7, p8]).ravel()  # (N,)

    # --- 파일별 min-max로 원단위 복원 ---
    cap_raw = load_csv(csv_path)[:, COL_CAP].astype(float)
    cap_min, cap_max = cap_raw.min(), cap_raw.max()
    pred_raw = pred_scaled * (cap_max - cap_min) + cap_min     # (N,)
    # true_raw = cap_raw[SEQ:]  # 필요하면 활성화

    # --- 전체 길이로 정렬(시퀀스 오프셋 보정) ---
    full_pred = np.full_like(cap_raw, np.nan, dtype=float)
    full_pred[SEQ:] = smooth(pred_raw, win=11)

    cycles = np.arange(len(cap_raw))

    # --- 플롯 ---
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(cycles, cap_raw,  "k-", label="True")
    ax.plot(cycles, full_pred,"g--",label="Pred")
    ax.axvline(x=SEQ, color="r", ls="--", lw=1.2)
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Discharge Capacity (Ah)")
    ax.set_title(csv_path.name)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_png = SAVE_PLOT_DIR / f"pred_{csv_path.stem}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png))
    plt.close(fig)
    print(f"✔ saved  {out_png}")

print("✔ Done: plots for selected test files (53,57,59).")
