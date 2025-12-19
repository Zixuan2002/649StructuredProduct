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

N = 1e6


# MC simulation parameters
T = 2.0       # 时间跨度：2年
M = 10000       # 模拟路径数量
s0 = [439.58, 491.02, 666.8]
#use the correlation matrix generated
corr_matrix = np.array([
    [1.00, 0.4097, 0.40], # TSLA 与其他
    [0.45, 1.00, 0.60], # MSFT 与其他
    [0.40, 0.60, 1.00]  # META 与其他
])
mu = [0.04, 0.04, 0.04]
sigma = [0.5105, 0.233, 0.322]
tsla, msft, meta = monte_carlo_antithetic(s0, mu, sigma, corr_matrix, T, M)

import numpy as np

def option_pricing(s1, s2, s3, K, T, bar1, bar2, c, N, r):
    """
    s1,s2,s3: each shape (M, steps+1) paths for TSLA/MSFT/META (or whatever order you use)
    K: strike ratio (e.g. 0.80)
    bar1, bar2: KO barriers for phase1/phase2 (e.g. 1.20, 1.10)
    c: coupon rate p.a. (e.g. 0.10)
    N: notional (e.g. 1e6)
    r: risk-free rate
    Returns: array of PVs, shape (M,)
    """
    M = s1.shape[0]
    payoffs = np.zeros(M)

    for j in range(M):
        path1 = s1[j]
        path2 = s2[j]
        path3 = s3[j]

        # Phase 1: indices 0..251 (1 year)
        p0 = np.minimum.reduce([
            path1[0:252] / path1[0],
            path2[0:252] / path2[0],
            path3[0:252] / path3[0]
        ])

        # Phase 2: indices 252..end (reset at 252)
        p1 = np.minimum.reduce([
            path1[252:] / path1[252],
            path2[252:] / path2[252],
            path3[252:] / path3[252]
        ])

        awarded = False

        # -------- Phase 1 KO: redeem immediately on day i --------
        for i in range(1, len(p0)):
            if p0[i] >= bar1:
                tau = i / 252.0  # KO time in years (immediate)
                pv = 0.0

                # coupons paid on scheduled quarter dates <= tau
                # (no stub; quarter dates are 0.25,0.5,0.75,1.0,...)
                for l in range(1, 9):
                    t_coupon = l / 4.0
                    if t_coupon <= tau:
                        pv += np.exp(-r * t_coupon) * (c / 4.0) * N

                # redemption at KO time tau
                pv += np.exp(-r * tau) * N
                payoffs[j] = pv
                awarded = True
                break

        # -------- Phase 2 KO: redeem immediately on day 1y + i --------
        if not awarded:
            for i in range(1, len(p1)):
                if p1[i] >= bar2:
                    tau = 1.0 + i / 252.0
                    pv = 0.0
                    for l in range(1, 9):
                        t_coupon = l / 4.0
                        if t_coupon <= tau:
                            pv += np.exp(-r * t_coupon) * (c / 4.0) * N
                    pv += np.exp(-r * tau) * N
                    payoffs[j] = pv
                    awarded = True
                    break

        # -------- No KO: maturity payoff --------
        if not awarded:
            pv = 0.0
            # all 8 coupons
            for l in range(1, 9):
                pv += np.exp(-r * (l / 4.0)) * (c / 4.0) * N

            P_T = p1[-1]  # phase-2 worst-of performance at maturity (for the K test)

            if P_T >= K:
                pv += np.exp(-r * T) * N
            else:
                # exact "worst stock" settlement relative to ISSUE DATE initial
                ratios_to_issue = np.array([
                    path1[-1] / path1[0],
                    path2[-1] / path2[0],
                    path3[-1] / path3[0]
                ])
                j_star = int(np.argmin(ratios_to_issue))

                S0_star = np.array([path1[0], path2[0], path3[0]])[j_star]
                ST_star = np.array([path1[-1], path2[-1], path3[-1]])[j_star]

                shares = N / (K * bar1 * S0_star)
                pv += np.exp(-r * T) * shares * ST_star

            payoffs[j] = pv

    return payoffs



#---setup---
K = 0.8
bar1 = 1.2
bar2 = 1.1
c = 0.1
r = 0.04
payoff_mc = option_pricing(tsla,meta,msft, K, T,bar1,bar2,c, N, r)
# print(f'Option Payoff: {payoff}')
print(f'Option Price from Monte Carlo: {np.mean(payoff_mc)}')
print(f'std error: {np.std(payoff_mc)/np.sqrt(len(payoff_mc))}')


# print(f'Option Price from Antithetic Variates Monte Carlo: {np.mean(payoff_antithetic)}')

# add a plot to visualize some of the simulated paths
# add a write-out to a .txt file to save results
# 在同一页上为每个标的绘制多条模拟路径（每个子图一只标的）并保存图片
'''
N_steps = tsla.shape[1] - 1
t = np.linspace(0, T, N_steps + 1)

# 要绘制的样本路径数量（最多绘制 M 条）
n_plot = min(10, tsla.shape[0])

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for i in range(n_plot):
    axes[0].plot(t, tsla[i], color='C0', alpha=0.5)
axes[0].plot(t, tsla.mean(axis=0), color='C0', linewidth=2, label='TSLA mean')
axes[0].set_title('TSLA simulated price paths')
axes[0].legend()

for i in range(n_plot):
    axes[1].plot(t, meta[i], color='C1', alpha=0.5)
axes[1].plot(t, meta.mean(axis=0), color='C1', linewidth=2, label='META mean')
axes[1].set_title('META simulated price paths')
axes[1].legend()

for i in range(n_plot):
    axes[2].plot(t, msft[i], color='C2', alpha=0.5)
axes[2].plot(t, msft.mean(axis=0), color='C2', linewidth=2, label='MSFT mean')
axes[2].set_title('MSFT simulated price paths')
axes[2].set_xlabel('Years')
axes[2].legend()

plt.tight_layout()
# plt.savefig('simulated_paths.png', dpi=150)
plt.show()
'''


