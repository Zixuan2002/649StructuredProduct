import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_n_assets(S0, mu, sigma, corr_matrix, T, M):
    """
    模拟 N 个相关资产的几何布朗运动路径
    
    参数:
    S0: 初始价格列表 [S0_1, S0_2, ..., S0_n]
    mu: 预期收益率列表 [mu_1, mu_2, ..., mu_n]
    sigma: 波动率列表 [sigma_1, sigma_2, ..., sigma_n]
    corr_matrix: N x N 相关系数矩阵 (numpy array)
    T: 时间长度 (年)
    M: 模拟路径数量 (Monte Carlo Simulations)
    
    返回:
    S: 形状为 (N_assets, M, Steps+1) 的数组
    """
    n_assets = len(S0)
    dt = 1/252
    steps = int(T * 252)
    
    # 1. Cholesky 分解: 找到矩阵 L，使得 L * L.T = Correlation Matrix
    # 这是将独立随机数转换为相关随机数的关键步骤
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("相关系数矩阵不是正定的 (Positive Definite)，无法进行 Cholesky 分解。请检查矩阵数值。")

    # 2. 生成不相关的随机数
    # 形状: (Assets, Simulations, Steps)
    Z_uncorr = np.random.standard_normal(size=(n_assets, M, steps))
    
    # 3. 引入相关性
    # 我们需要对每个时间步、每条路径的随机向量应用 L 变换
    # Z_corr[i, m, t] = sum(L[i, k] * Z_uncorr[k, m, t])
    # 使用 einsum 进行高效的张量乘法: 'ij, jms -> ims'
    # i, j: L矩阵行列; m: 模拟路径; s: 时间步
    Z_corr = np.einsum('ij,jms->ims', L, Z_uncorr)
    
    # 4. 模拟路径 (GBM)
    # 初始化价格数组 (Assets, M, Steps+1)
    S = np.zeros((n_assets, M, steps + 1))
    
    # 设置初始价格
    for i in range(n_assets):
        S[i, :, 0] = S0[i]
        
    # 计算每一时间步
    # S_t = S_{t-1} * exp(...)
    # 为了速度，我们直接计算 daily_returns 矩阵然后累乘
    
    for i in range(n_assets):
        drift = (mu[i] - 0.5 * sigma[i]**2) * dt
        diffusion = sigma[i] * np.sqrt(dt) * Z_corr[i]
        daily_returns = np.exp(drift + diffusion)
        
        # 累乘得到路径
        S[i, :, 1:] = S0[i] * np.cumprod(daily_returns, axis=1)
        
    return S

# def monte_carlo_gbm_antithetic(s0, mu, sigma, T, M):

#     dt = 1/252
#     N = int(T * 252)
    
#     # 1. 确保 M 是偶数，因为我们要成对生成
#     if M % 2 != 0:
#         M += 1
    
#     # 2. 只生成一半的随机数 (M/2)
#     Z_half = np.random.standard_normal(size=(M // 2, N))
    
#     # 3. 构造对偶变量：一半是 Z，另一半是 -Z
#     # 这样能保证这一批随机数的样本均值严格为 0，偏度也为 0
#     Z = np.concatenate((Z_half, -Z_half), axis=0)
    
#     drift = (mu - 0.5 * sigma**2) * dt
#     diffusion = sigma * np.sqrt(dt) * Z
    
#     daily_returns = np.exp(drift + diffusion)
    
#     S = np.zeros((M, N + 1))
#     S[:, 0] = s0
#     S[:, 1:] = s0 * np.cumprod(daily_returns, axis=1)
    
#     return S


N = 1e6


# MC simulation parameters
T = 2.0       # 时间跨度：2年
M = 1000       # 模拟路径数量
s0 = [439.58, 491.02, 666.8]
corr_matrix = np.array([
    [1.00, 0.45, 0.40], # TSLA 与其他
    [0.45, 1.00, 0.60], # MSFT 与其他
    [0.40, 0.60, 1.00]  # META 与其他
])
mu = [0.04, 0.04, 0.04]
sigma = [0.5105, 0.233, 0.322]
tsla, msft, meta = monte_carlo_n_assets(s0, mu, sigma, corr_matrix, T, M)

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
# payoff_antithetic = option_pricing(
#     monte_carlo_gbm_antithetic(S0, mu, sigma, T, M),
#     monte_carlo_gbm_antithetic(S1, mu, sigma1, T, M),
#     monte_carlo_gbm_antithetic(S2, mu, sigma2, T, M),
#     K, T, bar1, bar2, c,N)  
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


