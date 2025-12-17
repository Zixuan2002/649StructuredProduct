import numpy as np
scenarios = {
    "base case": {"vol":[0.5105,0.233,0.322], "corr":np.array([[1.00, 0.45, 0.40], [0.45, 1.00, 0.60], [0.40, 0.60, 1.00]  ])},
    "correlation risk": {"vol":[0.5105,0.233,0.322], "corr":np.array([[1.00, 0.9, 0.9], [0.9, 1.00, 0.9], [0.9, 0.90, 1.00]  ])},
    "correlation risk2": {"vol":[0.5105,0.233,0.322], "corr":np.array([[1.00, 0.1, 0.1], [0.1, 1.00, 0.1], [0.1, 0.1, 1.00]  ])},
    "volatility risk": {"vol":[0.7,0.4,0.5], "corr":np.array([[1.00, 0.45, 0.40], [0.45, 1.00, 0.60], [0.40, 0.60, 1.00]  ])},
    "collapse risk": {"vol":[0.5105,0.233,0.322], "corr":np.array([[1.00, 0.45, 0.40], [0.45, 1.00, 0.60], [0.40, 0.60, 1.00]  ]), "shock":1}
}

    
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

#---setup---
K = 0.8
bar1 = 1.2
bar2 = 1.1
c = 0.1
r = 0.04
T = 2.0       
M = 10000       
s0 = [439.58, 491.02, 666.8]
mu = [0.04, 0.04, 0.04]
N = 1e6

for scenario in scenarios:
    vol = scenarios[scenario]["vol"]
    corr_matrix = scenarios[scenario]["corr"]
    shock = scenarios[scenario].get("shock", 0)
    path1, path2, path3 = monte_carlo_antithetic(s0, mu, vol, corr_matrix, T, M)
    if shock:
        path1[:,300:] *= 0.6  # apply shock to TSLA paths
    payoff_mc = option_pricing(path1,path2,path3, K, T,bar1,bar2,c,N)
    print(f'Scenario: {scenario}, Option Price from Monte Carlo: {np.mean(payoff_mc)}')
    print(f'std error: {np.std(payoff_mc)/np.sqrt(len(payoff_mc))}')
