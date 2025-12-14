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
# --- tsla ---
S0 = 439.58      # 初始股价
mu = 0.04     # interest rate 4%
sigma = 0.5105  # 年化波动率 51.05%
T = 2.0       # 时间跨度：2年
M = 100       # 模拟路径数量：100条 (正式运行时通常设为 10000+)

# --- meta ---
S1 = 666.8
sigma1 = 0.322
# --- msft ---
S2 = 491.02
sigma2 = 0.233

tsla = monte_carlo_gbm(S0, mu, sigma, T, M)
meta = monte_carlo_gbm(S1, mu, sigma1, T, M)
msft = monte_carlo_gbm(S2, mu, sigma2, T, M)

def option_pricing(s1, s2, s3, K, T, bar1, bar2, c):
    

    M = s1.shape[0]
    payoffs = np.zeros(M)

    for j in range(M):
        path1 = s1[j]
        path2 = s2[j]
        path3 = s3[j]

        p0 = np.minimum.reduce([path1[0:252] / path1[0],
                                path2[0:252] / path2[0],
                                path3[0:252] / path3[0]])
        p1 = np.minimum.reduce([path1[252:] / path1[252],
                                path2[252:] / path2[252],
                                path3[252:] / path3[252]])

        awarded = False
        for i in range(len(p0)):
            if p0[i] >= bar1:
                payoffs[j] = 1 + (i // 63) * c / 4
                awarded = True
                break

        if not awarded:
            for i in range(len(p1)):
                if p1[i] >= bar2:
                    payoffs[j] = 1 + ((i + 252) // 63) * c / 4
                    awarded = True
                    break

        if not awarded:
            if p0.min() >= K and p1.min() >= K:
                payoffs[j] = 1 + c * T
            else:
                payoffs[j] = p1[-1]

    return payoffs


#---setup---
K = 0.8
bar1 = 1.2
bar2 = 1.1
c = 0.1
r = 0.04
payoff = option_pricing(tsla,meta,msft, K, T,bar1,bar2,c)*1e6
# print(f'Option Payoff: {payoff}')

def pricing(s1,s2,s3,r):
    payoff = option_pricing(s1,s2,s3, K, T,bar1,bar2,c)
    discounted_payoff = np.exp(-r * T) * payoff*1e6
    price = np.mean(discounted_payoff)  
    return price

option_price = pricing(tsla,meta,msft,r)    
print(f'Option Price: {option_price:.2f}')