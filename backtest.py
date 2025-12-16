import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_and_segment_by_quarter(csv_file_path):
    if not os.path.exists(csv_file_path):
        print(f"Can not find file: {csv_file_path}")
        return None

    df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    prices = df[price_col]

    print(f"正在处理: {csv_file_path} (使用列: {price_col})")
    quarterly_arrays = []
    df.index = pd.to_datetime(df.index)
    for period, group_data in prices.groupby(pd.Grouper(freq='Q')):
        if not group_data.empty:
            arr = group_data.to_numpy()
            quarterly_arrays.append((period.date(), arr))

    return quarterly_arrays

tsla_path = "C:\\Users\\snake\\TSLA_10y.csv" # 确保当前目录下有这个文件
msft_path = "C:\\Users\\snake\\MSFT_10y.csv"
meta_path = "C:\\Users\\snake\\META_10y.csv"
tsla_segments = load_and_segment_by_quarter(tsla_path)
msft_segments = load_and_segment_by_quarter(msft_path)
meta_segments = load_and_segment_by_quarter(meta_path)

K = 0.8
bar1 = 1.2
bar2 = 1.1
c = 0.1
r = 0.04
T = 2
N = 1e6
payoffs = np.zeros(33)
awarded = False
# if tsla_segments:
#     print(f"\n共分割出 {len(tsla_segments)} 个季度数据。\n")
    
#     # 打印前 3 个季度的情况看看
#     for i in range(len(tsla_segments)):
#         q_date, data = tsla_segments[i]
#         print(f"季度结束日: {q_date}")
#         print(f"数据形状: {data.shape} (即该季度有多少个交易日)")
#         print("-" * 30)

for i in range(1,34):
    tsla00 = tsla_segments[i][1][0]
    meta00 = meta_segments[i][1][0]
    msft00 = msft_segments[i][1][0]    
    for j in range(4): 
        data0 = tsla_segments[i+j][1]
        data1 = meta_segments[i+j][1]
        data2 = msft_segments[i+j][1]
        if np.minimum.reduce([data0/tsla00, data1/meta00, data2/msft00]).max() >= bar1 and not awarded:
            for k in range(1,j+1):
               payoffs[i-1] += np.exp(-r * (k/4)) * c / 4
            awarded = True
            payoffs[i-1] += np.exp(-r * ( (j+1)/4))
            break
    for j in range(4):
        data0 = tsla_segments[i+4+j][1]
        data1 = meta_segments[i+4+j][1]
        data2 = msft_segments[i+4+j][1]
        if np.minimum.reduce([data0/meta00, data1/meta00, data2/msft00]).max() >= bar2 and not awarded:
            for k in range(1,j+5):
               payoffs[i-1] += np.exp(-r * (k/4)) * c / 4
            awarded = True
            payoffs[i-1] += np.exp(-r * ( (j+5)/4))
            break
    if not awarded:
        if min(tsla_segments[i+7][1][-1]/tsla00, meta_segments[i+7][1][-1]/meta00, msft_segments[i+7][1][-1]/msft00) > K:
            payoffs[i-1] += np.exp(-r * T)
            for k in range(1,9):
                payoffs[i-1] += np.exp(-r * (k/4)) * c / 4
        else:
            payoffs[i-1] += (min(tsla_segments[i+7][1][-1]/tsla00, meta_segments[i+7][1][-1]/meta00, msft_segments[i+7][1][-1]/msft00))/(K*bar1)
            for k in range(1,9):
                payoffs[i-1] += np.exp(-r * (k/4)) * c / 4
    payoffs[i-1] = payoffs[i-1] * N
    awarded = False
#print(f'Option Payoff: {payoffs}')
            
print(f'Average Option Payoff over ten years: {np.mean(payoffs)}')
print(f'standard deviation of Option Payoff over ten years: {np.std(payoffs)}')

# 绘制 payoffs 的直方图并保存
try:
    plt.figure(figsize=(8, 5))
    plt.hist(payoffs, bins=12, edgecolor='black', alpha=0.75)
    plt.title('Histogram of Option Payoffs')
    plt.xlabel('Payoff')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.6)
    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), 'payoffs_histogram.png')
    plt.savefig(out_path)
    print(f'Saved histogram to {out_path}')
    plt.show()
except Exception as e:
    print('Could not display/save histogram:', e)


