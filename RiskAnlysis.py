#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.optimize import brentq
from montecarlo import monte_carlo_antithetic
from functions import option_pricing_fd


# In[3]:


def monte_carlo_antithetic_crn(S0, mu, sigma, corr_matrix, T, M, Z_half=None, seed=123):
    """
    Same as your monte_carlo_antithetic, but:
    - takes Z_half as an input to enable CRN for finite differences
    - returns (S, Z_half_used)
    S shape: (n_assets, M, steps+1)
    Z_half shape: (n_assets, M//2, steps)
    """
    S0 = np.asarray(S0, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)

    n_assets = len(S0)
    dt = 1/252
    steps = int(T * 252)

    # Cholesky
    L = np.linalg.cholesky(corr_matrix)

    # Generate or reuse Z_half
    half = M // 2
    if Z_half is None:
        rng = np.random.default_rng(seed)
        Z_half = rng.standard_normal(size=(n_assets, half, steps))

    # Antithetic completion (exactly your logic)
    Z_uncorr = np.concatenate((Z_half, -Z_half), axis=1)   # (n_assets, M, steps) if M even
    if Z_uncorr.shape[1] < M:  # if M odd, pad one path
        Z_uncorr = np.concatenate((Z_uncorr, Z_uncorr[:, :1, :]), axis=1)
    Z_uncorr = Z_uncorr[:, :M, :]

    # Correlate across assets
    Z_corr = np.einsum('ij,jms->ims', L, Z_uncorr)  # (n_assets, M, steps)

    # Build paths
    S = np.zeros((n_assets, M, steps + 1))
    for i in range(n_assets):
        S[i, :, 0] = S0[i]
        drift = (mu[i] - 0.5 * sigma[i]**2) * dt
        diffusion = sigma[i] * np.sqrt(dt) * Z_corr[i]         # (M, steps)
        daily_returns = np.exp(drift + diffusion)              # (M, steps)
        S[i, :, 1:] = S0[i] * np.cumprod(daily_returns, axis=1)

    return S, Z_half


# In[4]:


def price_mc_from_spot(S0, mu, sigma, corr_matrix, T, M, K, bar1, bar2, c, N, r, Z_half=None, seed=123):
    S, Z_used = monte_carlo_antithetic_crn(S0, mu, sigma, corr_matrix, T, M, Z_half=Z_half, seed=seed)
    S_origin = np.array([439.58, 491.02, 666.8])
    # IMPORTANT: your option_pricing expects paths shaped (M, steps+1)
    # Your S is (n_assets, M, steps+1), so pass each asset as S[i]
    payoffs = option_pricing_fd(S[0], S[1], S[2], K, T, bar1, bar2, c, N, r, S_origin)

    pv = float(np.mean(payoffs))
    se = float(np.std(payoffs, ddof=1) / np.sqrt(len(payoffs)))
    return pv, se, Z_used

# In[5]:


def fd_delta_gamma_spot_per_stock(
    S0, mu, sigma, corr_matrix, T, M, K, bar1, bar2, c, N, r,
    hS=None, seed=123
):
    S0 = np.asarray(S0, float)
    n = len(S0)

    if hS is None:
        hS = np.full(n, 0.005)  # default 0.5% bump each
    else:
        hS = np.asarray(hS, float)
        if hS.shape != (n,):
            raise ValueError(f"hS must have shape ({n},), got {hS.shape}")

    # Base price and frozen randomness
    V0, se0, Z_half = price_mc_from_spot(S0, mu, sigma, corr_matrix, T, M, K, bar1, bar2, c, N, r, Z_half=None, seed=seed)

    deltas = np.zeros(n)
    gammas = np.zeros(n)

    for i in range(n):
        dS = S0[i] * hS[i]   # absolute bump

        S_up = S0.copy(); S_up[i] += dS
        S_dn = S0.copy(); S_dn[i] -= dS

        V_up, _, _ = price_mc_from_spot(S_up, mu, sigma, corr_matrix, T, M, K, bar1, bar2, c, N, r, Z_half=Z_half, seed=seed)
        V_dn, _, _ = price_mc_from_spot(S_dn, mu, sigma, corr_matrix, T, M, K, bar1, bar2, c, N, r, Z_half=Z_half, seed=seed)

        # Delta_i = dV/dS_i
        deltas[i] = (V_up - V_dn) / (2.0 * dS)

        # Gamma_i = d^2V/dS_i^2
        gammas[i] = (V_up - 2.0 * V0 + V_dn) / (dS ** 2)

    return {
        "PV": V0,
        "PV_se": se0,
        "delta": deltas,
        "gamma": gammas
    }


# In[6]:


corr_matrix = np.array([
    [1.00, 0.4097, 0.3254], # TSLA 与其他
    [0.4097, 1.00, 0.5848], # MSFT 与其他
    [0.3254, 0.5848, 1.00]  # META 与其他
])

res = fd_delta_gamma_spot_per_stock(
    S0=[439.58, 491.02, 666.8],
    mu=[0.04, 0.04, 0.04],
    sigma=[0.5105, 0.233, 0.322],
    corr_matrix=corr_matrix,
    T=2.0, M=10000,
    K=0.8, bar1=1.2, bar2=1.1, c=0.1,
    N=1e6, r=0.04,
    hS=[0.01, 0.01, 0.01],  # per-stock bumps
    seed=123
)

print("PV:", res["PV"], "SE:", res["PV_se"])
print("Delta (TSLA, MSFT, META):", res["delta"])
print("Gamma (TSLA, MSFT, META):", res["gamma"])

