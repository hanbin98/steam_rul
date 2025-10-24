# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:52:13 2025

@author: chohb
"""

import os
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
PLOT_DIR = os.path.join('plot')
os.makedirs(PLOT_DIR, exist_ok=True)


def smooth_ma(x, k=5):

    return pd.Series(x).rolling(window=k, min_periods=1, center=True).mean().to_numpy()


def model_func(x, a, b, c, d):
    v, T, N, I = x
    return 2 - (a*T*N)*jnp.exp(d*I) - jnp.exp(b*(v-1.5)) * jnp.exp(c*N)


all_Qd, all_N, all_v, all_T, all_I = [], [], [], [], []

for i in range(1, 60):
    fname = f'B00{i}_capacity_raw.csv'
    fname = os.path.join('data', f'B00{i}_capacity_raw.csv')
    if not os.path.exists(fname):
        continue

    bat = pd.read_csv(fname, sep=',', header=None).to_numpy()

    Qd_raw = bat[:, 1].astype(float)
    Qd_max = Qd_raw.max() if Qd_raw.size else 1.0
    Qd     = Qd_raw / (Qd_max if Qd_max != 0 else 1.0)

    N = bat[:, 0].astype(float)
    v = bat[:, 4].astype(float)
    T = bat[:, 2].astype(float)
    I = bat[:, 3].astype(float)

    all_Qd.extend(Qd)
    all_N.extend(N)
    all_v.extend(v)
    all_T.extend(T)
    all_I.extend(I)


def model(v, T, N, I, Qd=None):
    a_raw = numpyro.sample('a_raw', dist.Normal(0.0, 2.0))
    a = numpyro.deterministic('a', jax.nn.softplus(a_raw))
    b = numpyro.sample('b', dist.Normal(0.0, 2.0))
    c = numpyro.sample('c', dist.Normal(0.0, 2.0))
    d = numpyro.sample('d', dist.Normal(0.0, 2.0))

    predicted_Qd = 2 - (a*T*N)*jnp.exp(d*I) - jnp.exp(b*(v-1.5)) * jnp.exp(c*N)
    numpyro.sample('obs', dist.Normal(predicted_Qd, 1), obs=Qd)

nuts_kernel = NUTS(model)
rng_key = jax.random.PRNGKey(0)
numpyro.set_host_device_count(2)

mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=200, num_chains=1)
mcmc.run(
    rng_key,
    jnp.array(all_v), jnp.array(all_T), jnp.array(all_N), jnp.array(all_I),
    Qd=jnp.array(all_Qd)
)

samples = mcmc.get_samples()
print("\nSummary of posterior samples:")
print(f"Posterior Mean - a: {jnp.mean(samples['a']):.4f}, b: {jnp.mean(samples['b']):.4f}, c: {jnp.mean(samples['c']):.4f}, d : {jnp.mean(samples['d']):.4f}")
print(f"Posterior Std  - a: {jnp.std(samples['a']):.4f}, b: {jnp.std(samples['b']):.4f}, c: {jnp.std(samples['c']):.4f}, d: {jnp.std(samples['d']):.4f}")


errors_per_batt = []
points_per_batt = []
all_true = []
all_pred = []


for i in range(1, 60):
    fname = f'B00{i}_capacity_raw.csv'
    fname = os.path.join('data', f'B00{i}_capacity_raw.csv')
    if not os.path.exists(fname):
        continue

    bat = pd.read_csv(fname, sep=',', header=None).to_numpy()

    Qd_raw = bat[:, 1].astype(float)
    Qd_sm  = smooth_ma(Qd_raw, k=5)
    Qd_max = Qd_sm.max() if Qd_sm.size else 1.0
    Qd     = Qd_sm / (Qd_max if Qd_max != 0 else 1.0)

    N = bat[:, 0].astype(float)
    v = bat[:, 4].astype(float)
    T = bat[:, 2].astype(float)
    I = bat[:, 3].astype(float)


    predicted_Qd_samples = []
    for a_s, b_s, c_s, d_s in zip(samples['a'], samples['b'], samples['c'], samples['d']):
        predicted_Qd_samples.append(
            model_func((jnp.array(v), jnp.array(T), jnp.array(N), jnp.array(I)), a_s, b_s, c_s, d_s)
        )
    predicted_Qd_samples = jnp.array(predicted_Qd_samples)

    predicted_Qd_mean     = jnp.mean(predicted_Qd_samples, axis=0)
    predicted_Qd_variance = jnp.var(predicted_Qd_samples, axis=0)


    pred_mean_rescaled = predicted_Qd_mean * Qd_max
    pred_std_rescaled  = jnp.sqrt(predicted_Qd_variance) * Qd_max


    denom = jnp.where(Qd_sm == 0, 1e-9, Qd_sm)
    err_pct = jnp.abs((Qd_sm - pred_mean_rescaled) / denom) * 100
    avg_err = jnp.mean(err_pct)
    print(f"Battery B00{i} - Avg Error (vs smoothed): {avg_err:.2f}%")
    
    errors_per_batt.append(avg_err)
    points_per_batt.append(int(len(N)))
    all_true.append(jnp.asarray(Qd_sm))
    all_pred.append(jnp.asarray(pred_mean_rescaled))


    fig = plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(N, Qd_sm, label='Measured (smoothed)', lw=3)
    plt.plot(N, pred_mean_rescaled, label='Predicted', lw=3)
    plt.fill_between(
        N,
        pred_mean_rescaled - 1.96 * pred_std_rescaled,
        pred_mean_rescaled + 1.96 * pred_std_rescaled,
        alpha=0.3, label='95% CI', color='tab:orange'
    )
    plt.xlabel("Cycle number", fontsize=15, labelpad=10)
    plt.ylabel("Discharging capacity", fontsize=15, labelpad=10)
    plt.ylim(0, 2.1)
    plt.title(f'Battery B00{i} - v: {v[0]}V, T: {T[0]}C')

    out_path = os.path.join(PLOT_DIR, f'B00{i}_pred.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✔ saved: {out_path}")
    
if errors_per_batt:
    macro_avg = sum(errors_per_batt) / len(errors_per_batt)
    micro_avg = (sum(e * n for e, n in zip(errors_per_batt, points_per_batt))
                 / sum(points_per_batt))


    y_true = jnp.concatenate(all_true) if all_true else jnp.array([])
    y_pred = jnp.concatenate(all_pred) if all_pred else jnp.array([])
    if y_true.size > 1:
        sst = jnp.sum((y_true - jnp.mean(y_true))**2)
        sse = jnp.sum((y_true - y_pred)**2)
        r2  = float(1.0 - (sse / sst)) if float(sst) > 0 else float('nan')
        print(f"\n[Summary] Mean error (macro): {macro_avg:.2f}%")
        print(f"[Summary] Mean error (micro): {micro_avg:.2f}%")
        print(f"[Summary] R^2 across all points: {r2:.4f}")
    else:
        print(f"\n[Summary] Mean error (macro): {macro_avg:.2f}%")
        print(f"[Summary] Mean error (micro): {micro_avg:.2f}%")
        print("[Summary] R^2 across all points: n/a (insufficient points)")
else:
    print("\n[Summary] No batteries processed.")


