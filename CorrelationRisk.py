#!/usr/bin/env python
# coding: utf-8

# ## Correlation Analysis in Python

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from montecarlo import monte_carlo_antithetic
from montecarlo import option_pricing
# * Data Preparation

# In[2]:


# Load 10-year OHLCV data files into DataFrames
df_META = pd.read_csv('META_10y.csv')
df_MSFT = pd.read_csv('MSFT_10y.csv')
df_TSLA = pd.read_csv('TSLA_10y.csv')


# In[3]:


# Option 1: Reassign each DataFrame
df_META = df_META.rename(columns={'Price': 'Date'})
df_META = df_META.iloc[2:]
df_META['Date'] = pd.to_datetime(df_META['Date'])
df_META.sort_values('Date', inplace=True)
df_META['Close'] = pd.to_numeric(df_META['Close'], errors='coerce')

df_MSFT = df_MSFT.rename(columns={'Price': 'Date'})
df_MSFT = df_MSFT.iloc[2:]
df_MSFT['Date'] = pd.to_datetime(df_MSFT['Date'])
df_MSFT.sort_values('Date', inplace=True)
df_MSFT['Close'] = pd.to_numeric(df_MSFT['Close'], errors='coerce')

df_TSLA = df_TSLA.rename(columns={'Price': 'Date'})
df_TSLA = df_TSLA.iloc[2:]
df_TSLA['Date'] = pd.to_datetime(df_TSLA['Date'])
df_TSLA.sort_values('Date', inplace=True)
df_TSLA['Close'] = pd.to_numeric(df_TSLA['Close'], errors='coerce')


# In[4]:


# Keep only Date and Adj Close
df_META = df_META[['Date', 'Close']].rename(columns={'Close': 'META'})
df_MSFT = df_MSFT[['Date', 'Close']].rename(columns={'Close': 'MSFT'})
df_TSLA = df_TSLA[['Date', 'Close']].rename(columns={'Close': 'TSLA'})

# Merge on common dates
df = df_META.merge(df_MSFT, on='Date', how='inner')
df = df.merge(df_TSLA, on='Date', how='inner')


# * Time Adjusted Log Returns

# In[5]:


# Compute time differences in days
df['dt'] = df['Date'].diff().dt.days 

# Compute log returns
for asset in ['META', 'MSFT', 'TSLA']:
    df[f'r_{asset}'] = np.log(df[asset] / df[asset].shift(1)) / np.sqrt(df['dt'])

# Drop first row (NaNs from diff)
df_returns = df.dropna(subset=['r_META', 'r_MSFT', 'r_TSLA'])
# * Correlation Matrix Calculation

# In[6]:


returns = df_returns[['r_META', 'r_MSFT', 'r_TSLA']].iloc[-255:-4]

corr_matrix = returns.corr()
corr_matrix.to_csv('corr 1y.txt', sep='\t', float_format='%.4f')
print("Correlation matrix saved to 'corr 1y.txt'.")

# * Estimate the error

# In[7]:


def fisher_ci(rho_hat : float , N : int, ci=0.95)-> float:
    """
    Fisher transformation confidence interval for correlation
    
    Parameters:
    -----------
    rho_hat : float
        Sample correlation coefficient
    N : int
        Sample size
    ci : float, optional (default=0.95)
        Confidence level (e.g., 0.95 for 95% CI, 0.99 for 99% CI)
    
    Returns:
    --------
    rho_low, rho_high : tuple
        Lower and upper bounds of the confidence interval
    """
    
    z = 0.5 * np.log((1 + rho_hat) / (1 - rho_hat))
    sigma_z = 1 / np.sqrt(N - 3)
    
    # Calculate z-critical value based on confidence level
    alpha = 1 - ci
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    z_low = z - z_crit * sigma_z
    z_high = z + z_crit * sigma_z
    
    rho_low = (np.exp(2*z_low) - 1) / (np.exp(2*z_low) + 1)
    rho_high = (np.exp(2*z_high) - 1) / (np.exp(2*z_high) + 1)
    
    
    return rho_low, rho_high


# In[8]:


N = len(returns)
ci = 0.95 # 95% confidence interval

# CI for correlation between META and MSFT
rho_META_MSFT = corr_matrix.iloc[1,0]
std_rho_META_MSFT = (1 - rho_META_MSFT**2)/np.sqrt(N-3)


# CI for correlation between META and TSLA
rho_META_TSLA = corr_matrix.iloc[2,0]
std_rho_META_TSLA = (1 - rho_META_TSLA**2)/np.sqrt(N-3)

# CI for correlation between TSLA and MSFT
rho_TSLA_MSFT = corr_matrix.iloc[2,1]
std_rho_TSLA_MSFT = (1 - rho_TSLA_MSFT**2)/np.sqrt(N-3)

# Calculate Alpha
alpha1 = std_rho_META_MSFT / (1-rho_META_MSFT)
alpha2 = std_rho_META_TSLA / (1-rho_META_TSLA)
alpha3 = std_rho_TSLA_MSFT / (1-rho_TSLA_MSFT)
alpha = (alpha1 + alpha2 + alpha3) / 3

# Move all the correlations up by alpha
rho_adj = corr_matrix + alpha*(1-corr_matrix)
with open('corr 1y.txt', 'a') as f:
    f.write(f"\nAdjusted correlation matrix:\n{rho_adj}")
print("Adjusted correlation matrix saved to 'corr 1y.txt'.")

# Estimate the impact from changed correlations
## Base case: original correlation matrix
N = 1e6
T = 2.0       # 时间跨度：2年
M = 100000       # 模拟路径数量
s0 = [439.58, 491.02, 666.8]
#use the correlation matrix generated
mu = [0.04, 0.04, 0.04]
sigma = [0.5105, 0.233, 0.322]
tsla, msft, meta = monte_carlo_antithetic(s0, mu, sigma, corr_matrix, T, M)
#---setup---
K = 0.8
bar1 = 1.2
bar2 = 1.1
c = 0.1
r = 0.04
payoff_mc = option_pricing(tsla,meta,msft, K, T,bar1,bar2,c, N, r)
# print(f'Option Payoff: {payoff}')
print(f'Price from Monte Carlo: {np.mean(payoff_mc)}')


# Estimate the impact from changed correlations
## Adjusted case: adjusted correlation matrix
N = 1e6
T = 2.0       # 时间跨度：2年
M = 100000       # 模拟路径数量
s0 = [439.58, 491.02, 666.8]
#use the correlation matrix generated
mu = [0.04, 0.04, 0.04]
sigma = [0.5105, 0.233, 0.322]
tsla, msft, meta = monte_carlo_antithetic(s0, mu, sigma, rho_adj, T, M)
#---setup---
K = 0.8
bar1 = 1.2
bar2 = 1.1
c = 0.1
r = 0.04
payoff_ad_mc = option_pricing(tsla,meta,msft, K, T,bar1,bar2,c, N, r)
# print(f'Option Payoff: {payoff}')
print(f'Adjusted Price from Monte Carlo: {np.mean(payoff_ad_mc)}')


delta_V = payoff_ad_mc - payoff_mc
print(f'Delta V: {np.mean(delta_V)}')
