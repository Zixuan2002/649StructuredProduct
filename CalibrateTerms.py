#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from scipy.optimize import brentq
from montecarlo import monte_carlo_antithetic
from montecarlo import option_pricing

# ---- helper: coupon schedule in year fractions (quarterly over 2y)
COUPON_TIMES = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])

def simulate_once(S0, mu, sigma, corr_matrix, T, M, seed=123):
    np.random.seed(seed)  # common random numbers
    S = monte_carlo_antithetic(S0, mu, sigma, corr_matrix, T, M)
    s1, s2, s3 = S[0], S[1], S[2]   # each (M, steps+1)
    return s1, s2, s3


def note_pv_given_K_immediateKO(s1, s2, s3, K, T, bar1, bar2, c, N, r):
    """
    PV of your note with:
      - daily KO monitoring
      - KO redeemed immediately on the KO day (discounted at that day)
      - coupons paid quarterly only if coupon date <= KO day (no stub accrual)
      - phase reset at 1y
      - maturity:
          if P_T >= K -> par
          else -> discounted N * (P_T / (K*bar1))  (consistent "conversion-style" downside)
    """
    M = s1.shape[0]
    payoffs = np.zeros(M)

    for j in range(M):
        path1, path2, path3 = s1[j], s2[j], s3[j]

        # Phase 1 ratios (t in [0,1y], indices 0..251)
        p0 = np.minimum.reduce([
            path1[0:252] / path1[0],
            path2[0:252] / path2[0],
            path3[0:252] / path3[0],
        ])

        # Phase 2 ratios (reset at index 252, t in [1y,2y], indices 252..end)
        p1 = np.minimum.reduce([
            path1[252:] / path1[252],
            path2[252:] / path2[252],
            path3[252:] / path3[252],
        ])

        # -------- KO check Phase 1 (immediate on day i) --------
        ko1 = np.where(p0 >= bar1)[0]
        if ko1.size > 0:
            i = int(ko1[0])                 # day index within year 1
            tau = i / 252.0                 # KO time in years

            # coupons paid only on coupon dates <= tau
            coupon_pv = np.sum(np.exp(-r * COUPON_TIMES[COUPON_TIMES <= tau]) * (c / 4) * N)

            # redeem immediately at par
            redemption_pv = np.exp(-r * tau) * N
            payoffs[j] = coupon_pv + redemption_pv
            continue

        # -------- KO check Phase 2 (immediate on day 252+i) --------
        ko2 = np.where(p1 >= bar2)[0]
        if ko2.size > 0:
            i = int(ko2[0])                 # day index within phase 2, i=0 corresponds to t=1y
            tau = 1.0 + i / 252.0           # KO time in years

            coupon_pv = np.sum(np.exp(-r * COUPON_TIMES[COUPON_TIMES <= tau]) * (c / 4) * N)
            redemption_pv = np.exp(-r * tau) * N
            payoffs[j] = coupon_pv + redemption_pv
            continue

        # -------- No KO: maturity payoff --------
        coupon_pv = np.sum(np.exp(-r * COUPON_TIMES) * (c / 4) * N)

        P_T = p1[-1]  # worst-of performance in phase 2 at maturity

        if P_T >= K:
            redemption_pv = np.exp(-r * T) * N
        else:
            redemption_pv = np.exp(-r * T) * N * (P_T / (K * bar1))

        payoffs[j] = coupon_pv + redemption_pv

    return float(np.mean(payoffs))


def calibrate_K(s1, s2, s3, T, bar1, bar2, c, N, r, K_low=0.60, K_high=0.95):
    def f(K):
        pv = option_pricing(s1, s2, s3, K, T, bar1, bar2, c, N, r).mean()
        return pv - N

    f_low, f_high = f(K_low), f(K_high)
    if f_low * f_high > 0:
        raise ValueError(f"Root not bracketed: f({K_low})={f_low}, f({K_high})={f_high}")

    K_star = brentq(f, K_low, K_high)
    pv_check = option_pricing(s1, s2, s3, K_star, T, bar1, bar2, c, N, r).mean()
    return K_star, pv_check


# -------------------------
# Run with your parameters
# -------------------------
N = 1e6
T = 2.0
M = 100000       # calibration needs more than 1000 paths
r = 0.04
c = 0.10
bar1, bar2 = 1.20, 1.10

s0 = [439.58, 491.02, 666.8]  # TSLA, MSFT, META in YOUR current order
corr_matrix = np.array([
    [1.00, 0.4097, 0.3254], # TSLA 与其他
    [0.4097, 1.00, 0.5848], # MSFT 与其他
    [0.3254, 0.5848, 1.00]  # META 与其他
])
mu = [0.04, 0.04, 0.04]
sigma = [0.5105, 0.233, 0.322]

s1, s2, s3 = simulate_once(s0, mu, sigma, corr_matrix, T, M, seed=123)

K_star, pv = calibrate_K(s1, s2, s3, T, bar1, bar2, c, N, r,
                                K_low=0.50, K_high=0.95)

print("Calibrated strike K* =", K_star)
print("PV check =", pv, " target =", N)


# In[31]:


def calibrate_B1(s1, s2, s3, K, T, B2, c, N, r,
                 B1_low=1.01, B1_high=1.60):
    """
    Solve for B1 such that E[PV(payoff)] = N, holding:
      - K fixed
      - B2 fixed
      - coupon fixed
    """
    def f(B1):
        pv = option_pricing(s1, s2, s3, K, T, B1, B2, c, N, r).mean()
        return pv - N

    f_low = f(B1_low)
    f_high = f(B1_high)
    if f_low * f_high > 0:
        raise ValueError(
            f"Root not bracketed on [{B1_low}, {B1_high}]. "
            f"f(B1_low)={f_low:.4f}, f(B1_high)={f_high:.4f}. "
            "Widen the bracket or reconsider whether PV is monotone in B1 for your payoff."
        )

    B1_star = brentq(f, B1_low, B1_high, xtol=1e-6, rtol=1e-10, maxiter=100)
    pv_check = option_pricing(s1, s2, s3, K, T, B1_star, B2, c, N, r).mean()
    return B1_star, pv_check


# In[37]:


# Fixed parameters
K = 0.80
B2 = 1.10  # per your request
T = 2.0
r = 0.04
c = 0.10
N = 1e6

# Calibrate B1
B1_star, pv = calibrate_B1(s1, s2, s3, K, T, B2, c, N, r,
                           B1_low=0.9, B1_high=1.2)

print("Calibrated B1* =", B1_star)
print("PV check       =", pv, " target =", N)


# In[38]:


print("p0[0] sample:", np.minimum.reduce([s1[0,0]/s1[0,0], s2[0,0]/s2[0,0], s3[0,0]/s3[0,0]]))
# should print 1.0


# In[39]:


def scan_B1(s1, s2, s3, K, T, B2, c, N, r, grid=None):
    if grid is None:
        grid = np.linspace(1.05, 1.80, 16)
    vals = []
    for B1 in grid:
        pv = option_pricing(s1, s2, s3, K, T, B1, B2, c, N, r).mean()
        vals.append((B1, pv - N))
    return vals

vals = scan_B1(s1, s2, s3, K, T, B2, c, N, r)
for B1, diff in vals:
    print(f"B1={B1:.3f}, PV-N={diff:,.2f}")


# In[42]:


def calibrate_B2(s1, s2, s3, K, T, B1, c, N, r,
                 B2_low=1.01, B2_high=1.80):
    """
    Solve for B2 such that E[PV(payoff)] = N, holding:
      - K fixed
      - B1 fixed
      - coupon fixed
    """
    def f(B2):
        pv = option_pricing(s1, s2, s3, K, T, B1, B2, c, N, r).mean()
        return pv - N

    f_low = f(B2_low)
    f_high = f(B2_high)
    if f_low * f_high > 0:
        raise ValueError(
            f"Root not bracketed on [{B2_low}, {B2_high}]. "
            f"f(B2_low)={f_low:.4f}, f(B2_high)={f_high:.4f}. "
            "Use scan_B2() to find a bracket."
        )

    B2_star = brentq(f, B2_low, B2_high, xtol=1e-6, rtol=1e-10, maxiter=100)
    pv_check = option_pricing(s1, s2, s3, K, T, B1, B2_star, c, N, r).mean()
    return B2_star, pv_check


# In[76]:


# Fixed terms
K = 0.65
B1 = 1.07
B2_initial_guess = 1.10  # just for intuition; calibration finds the real one

B2_star, pv = calibrate_B2(
    s1, s2, s3,
    K=K, T=2.0, B1=B1,
    c=0.10, N=1e6, r=0.04,
    B2_low=1.0, B2_high=1.60
)

print("Calibrated B2* =", B2_star)
print("PV check       =", pv, " target =", 1e6)


# In[81]:


def calibrate_coupon_linear(s1, s2, s3, K, T, B1, B2, N, r, c1=0.10):
    """
    Calibrate coupon c so that E[PV]=N, using linearity in c.
    Requires option_pricing(s1,s2,s3,K,T,B1,B2,c,N,r) returning PVs per path.
    """
    pv0 = option_pricing(s1, s2, s3, K, T, B1, B2, c=0.00, N=N, r=r).mean()
    pv1 = option_pricing(s1, s2, s3, K, T, B1, B2, c=c1, N=N, r=r).mean()

    if abs(pv1 - pv0) < 1e-9:
        raise ValueError("PV does not change with coupon in your implementation; check coupon logic.")

    c_star = c1 * (N - pv0) / (pv1 - pv0)
    pv_check = option_pricing(s1, s2, s3, K, T, B1, B2, c=c_star, N=N, r=r).mean()
    return c_star, pv0, pv1, pv_check


# In[83]:


# Original settings
K = 0.80
B1 = 1.20
B2 = 1.10
T = 2.0
r = 0.04
N = 1e6

# Use the SAME simulated paths for stability
# (i.e., s1,s2,s3 already from simulate_once with fixed seed)
c_star, pv0, pv10, pv_check = calibrate_coupon_linear(
    s1, s2, s3, K=K, T=T, B1=B1, B2=B2, N=N, r=r, c1=0.10
)

print("PV(c=0)   =", pv0)
print("PV(c=10%) =", pv10)
print("Calibrated coupon c* =", c_star)
print("PV check  =", pv_check, " target =", N)

