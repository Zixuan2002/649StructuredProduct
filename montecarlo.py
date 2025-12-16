import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_gbm(s0, mu, sigma, T, M):
    # 计算每个时间步的时长
    dt = 1/252
    N = int(T * 252)  
    Z = np.random.standard_normal(size=(M, N))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    daily_returns = np.exp(drift + diffusion)
    S = np.zeros((M, N + 1))
    S[:, 0] = s0

    # 使用 cumprod (累积乘积) 快速推算路径
    # S_t = S_0 * product(daily_returns)
    S[:, 1:] = s0 * np.cumprod(daily_returns, axis=1)

    #S = S.mean(axis=0)

    return S

import numpy as np

def monte_carlo_gbm_antithetic(s0, mu, sigma, T, M):

    dt = 1/252
    N = int(T * 252)
    
    # 1. 确保 M 是偶数，因为我们要成对生成
    if M % 2 != 0:
        M += 1
    
    # 2. 只生成一半的随机数 (M/2)
    Z_half = np.random.standard_normal(size=(M // 2, N))
    
    # 3. 构造对偶变量：一半是 Z，另一半是 -Z
    # 这样能保证这一批随机数的样本均值严格为 0，偏度也为 0
    Z = np.concatenate((Z_half, -Z_half), axis=0)
    
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    daily_returns = np.exp(drift + diffusion)
    
    S = np.zeros((M, N + 1))
    S[:, 0] = s0
    S[:, 1:] = s0 * np.cumprod(daily_returns, axis=1)
    
    return S
# --- tsla ---
S0 = 439.58      # 初始股价
mu = 0.04     # interest rate 4%
sigma = 0.5105  # 年化波动率 51.05%
T = 2.0       # 时间跨度：2年
M = 100       # 模拟路径数量
N = 1e6

# --- meta ---
S1 = 666.8
sigma1 = 0.322
# --- msft ---
S2 = 491.02
sigma2 = 0.233

tsla = monte_carlo_gbm(S0, mu, sigma, T, M)
meta = monte_carlo_gbm(S1, mu, sigma1, T, M)
msft = monte_carlo_gbm(S2, mu, sigma2, T, M)

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
                    payoffs[j] += np.exp(-r * (l/4)) * c / 4
            else:
                payoffs[j] = N/(K*bar1) * p1[-1]*p0[-1] # need to correct

    return payoffs


#---setup---
K = 0.8
bar1 = 1.2
bar2 = 1.1
c = 0.1
r = 0.04
payoff_mc = option_pricing(tsla,meta,msft, K, T,bar1,bar2,c,N)
# print(f'Option Payoff: {payoff}')
print(f'Option Price from Monte Carlo: {np.mean(payoff_mc)}')
payoff_antithetic = option_pricing(
    monte_carlo_gbm_antithetic(S0, mu, sigma, T, M),
    monte_carlo_gbm_antithetic(S1, mu, sigma1, T, M),
    monte_carlo_gbm_antithetic(S2, mu, sigma2, T, M),
    K, T, bar1, bar2, c,N)  
print(f'Option Price from Antithetic Variates Monte Carlo: {np.mean(payoff_antithetic)}')

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


