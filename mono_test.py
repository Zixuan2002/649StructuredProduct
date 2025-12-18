import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_antithetic(S0, mu, sigma, corr_matrix, T, M):
    n_assets = len(S0)
    dt = 1/252
    steps = int(T * 252)
    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Not positive definite correlation matrix.")


    Z_half = np.random.standard_normal(size=(n_assets, M//2, steps))
    Z_uncorr = np.concatenate((Z_half, -Z_half), axis=1)
    Z_corr = np.einsum('ij,jms->ims', L, Z_uncorr)
    
    S = np.zeros((n_assets, M, steps + 1))
    for i in range(n_assets):
        S[i, :, 0] = S0[i]
        
    
    for i in range(n_assets):
        drift = (mu[i] - 0.5 * sigma[i]**2) * dt
        diffusion = sigma[i] * np.sqrt(dt) * Z_corr[i]
        daily_returns = np.exp(drift + diffusion)
        S[i, :, 1:] = S0[i] * np.cumprod(daily_returns, axis=1)
        
    return S

def option_pricing(s1, s2, s3, K, T, bar1, bar2, c,N):
    

    M = s1.shape[0]
    payoffs = np.zeros(M)

    for j in range(M):
        path1 = s1[j]
        path2 = s2[j]
        path3 = s3[j]

        p0 = np.minimum.reduce([path1[0:252] / path1[0],
                                path2[0:252] / path2[0],
                                path3[0:252] / path3[0]]) # the path of P in year 1
        p1 = np.minimum.reduce([path1[252:] / path1[252],
                                path2[252:] / path2[252],
                                path3[252:] / path3[252]]) # the path of P in year 2

        awarded = False
        # for phase 1
        for i in range(len(p0)):
            if p0[i] >= bar1:
                k = i//63
                payoffs[j] = 0
                if k >= 1:
                    for l in range(1,k+1):
                        payoffs[j] += np.exp(-r * (l/4)) * c / 4
                payoffs[j] += np.exp(-r * (k/4))
                payoffs[j] = N * payoffs[j]
                awarded = True
                break
        # for phase 2
        if not awarded:
            for i in range(len(p1)):
                if p1[i] >= bar2:
                    k = i//63+4
                    payoffs[j] = 0
                    for l in range(1,k+1):
                        payoffs[j] += np.exp(-r * (l/4)) * c / 4
                    payoffs[j] += np.exp(-r * (k/4))
                    payoffs[j] = N * payoffs[j]
                    awarded = True
                    break

        if not awarded:
            if p1[-1] >= K:
                payoffs[j] = np.exp(-r * T) * N
                for l in range(1,9):
                    payoffs[j] += np.exp(-r * (l/4)) * c / 4 *N
            else:
                payoffs[j] = N/(K*bar1) * p1[-1]*p0[-1] # need to correct
                for l in range(1,9):
                    payoffs[j] += np.exp(-r * (l/4)) * c / 4 *N

    return payoffs




#---spot Monotonicity---
K = 0.8
bar1 = 1.2
bar2 = 1.1
c = 0.1
r = 0.04
T = 2.0       
M = 1000       
s0 = [439.58, 491.02, 666.8]
mu = [0.04, 0.04, 0.04]
N = 1e6
sigma = [0.5105, 0.233, 0.322]
corr_matrix = np.array([
    [1.00, 0.4097, 0.3254], # TSLA 与其他
    [0.4097, 1.00, 0.5848], # MSFT 与其他
    [0.3254, 0.5848, 1.00]  # META 与其他
])

ratio = np.linspace(0.5, 1.5, 11)
payoff_spot_mono = []
for s in ratio:
    tsla, msft, meta = monte_carlo_antithetic(s0, mu, sigma, corr_matrix, T, M)
    tsla[:,1:] *= s
    msft[:,1:] *= s
    meta[:,1:] *= s
    payoff_mc = option_pricing(tsla,meta,msft, K, T,bar1,bar2,c,N)
    payoff_spot_mono.append(np.mean(payoff_mc))
for s, p in zip(ratio, payoff_spot_mono):
    print(f'Spot Ratio: {s:.2f}, Option Price from Monte Carlo: {p:.2f}')

#--- volatility Monotonicity ---
alpha = np.linspace(-0.2,0.2,9)
payoff_vol_mono = []
for a in alpha:
    sigma_new = sigma+a
    tsla, msft, meta = monte_carlo_antithetic(s0, mu, sigma_new, corr_matrix, T, M)
    payoff_mc = option_pricing(tsla,meta,msft, K, T,bar1,bar2,c,N)
    payoff_vol_mono.append(np.mean(payoff_mc))
for a, p in zip(alpha, payoff_vol_mono):
    print(f'Volatility Adjustment: {a:.2f}, Option Price from Monte Carlo: {p:.2f}')

#---correlation Monotonicity---
corrs = np.linspace(0.1,0.9,9)
payoff_corr_mono = []
for corr in corrs:
    corr_matrix_new = np.array([
        [1.00, corr, corr], 
        [corr, 1.00, corr], 
        [corr, corr, 1.00]  
    ])
    tsla, msft, meta = monte_carlo_antithetic(s0, mu, sigma, corr_matrix_new, T, M)
    payoff_mc = option_pricing(tsla,meta,msft, K, T,bar1,bar2,c,N)
    payoff_corr_mono.append(np.mean(payoff_mc))
for c, p in zip(corrs, payoff_corr_mono):
    print(f'Correlation: {c:.2f}, Option Price from Monte Carlo: {p:.2f}')

# --- plot and save the three payoff monotonicity series ---
import os
os.makedirs('plots', exist_ok=True)

# spot monotonicity
plt.figure(figsize=(8,4))
plt.plot(ratio, payoff_spot_mono, marker='o')
plt.xlabel('Spot Ratio')
plt.ylabel('Option Price')
plt.title('Spot Monotonicity')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('plots','payoff_spot_monotonicity.png'), dpi=150)
plt.close()

# volatility monotonicity
plt.figure(figsize=(8,4))
plt.plot(alpha, payoff_vol_mono, marker='o')
plt.xlabel('Volatility Adjustment')
plt.ylabel('Option Price')
plt.title('Volatility Monotonicity')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('plots','payoff_vol_monotonicity.png'), dpi=150)
plt.close()

# correlation monotonicity
plt.figure(figsize=(8,4))
plt.plot(corrs, payoff_corr_mono, marker='o')
plt.xlabel('Correlation')
plt.ylabel('Option Price')
plt.title('Correlation Monotonicity')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('plots','payoff_corr_monotonicity.png'), dpi=150)
plt.close()

