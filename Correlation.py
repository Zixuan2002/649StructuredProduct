#!/usr/bin/env python
# coding: utf-8

# ## Correlation Analysis in Python

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


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


returns = df_returns[['r_META', 'r_MSFT', 'r_TSLA']]

corr_matrix = returns.corr()
corr_matrix.to_csv('corr.txt', sep='\t', float_format='%.4f')
print("Correlation matrix saved to 'corr.txt'.")


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
rho_low_META_MSFT, rho_high_META_MSFT = fisher_ci(rho_META_MSFT, N, ci)

# CI for correlation between META and TSLA
rho_META_TSLA = corr_matrix.iloc[2,0]
rho_low_META_TSLA, rho_high_META_TSLA = fisher_ci(rho_META_TSLA, N, ci)

# CI for correlation between TSLA and MSFT
rho_TSLA_MSFT = corr_matrix.iloc[2,1]
rho_low_TSLA_MSFT, rho_high_TSLA_MSFT = fisher_ci(rho_TSLA_MSFT, N, ci)

# write to .txt file I created earlier
with open('corr.txt', 'a') as f:
    f.write(f"\n{ci*100}% CI for correlation between META and MSFT: ({rho_low_META_MSFT:.4f}, {rho_high_META_MSFT:.4f})\n")
    f.write(f"{ci*100}% CI for correlation between META and TSLA: ({rho_low_META_TSLA:.4f}, {rho_high_META_TSLA:.4f})\n")
    f.write(f"{ci*100}% CI for correlation between TSLA and MSFT: ({rho_low_TSLA_MSFT:.4f}, {rho_high_TSLA_MSFT:.4f})\n")

print(f"{ci*100}% CI for correlation between META, MSFT and TSLA saved to 'corr.txt'.")

