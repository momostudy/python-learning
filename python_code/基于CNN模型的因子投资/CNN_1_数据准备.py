#%% ===========================
#%% Part 1: 数据准备
#%% ===========================

#%%
import numpy as np
import pandas as pd
# import sqlalchemy as sa
import os
os.chdir('D:\FinTech_2022') 
import configparser
import time
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import datetime
warnings.filterwarnings("ignore")

#%% 整理基准数据【已完成】
# 所有日期均调整为int，8位数字
# 数据来源：国泰安
# 导入000300收盘价
# FUND_IDX_QUOTATIONDAY-1.xlsx
# FUND_IDX_QUOTATIONDAY-2.xlsx

IDX_1 = pd.read_excel('FUND_IDX_QUOTATIONDAY-1.xlsx')
IDX_2 = pd.read_excel('FUND_IDX_QUOTATIONDAY-2.xlsx')
indexdata = pd.concat([IDX_1,IDX_2])
del indexdata['Symbol']
indexdata.columns = ['date','close']
indexdata = indexdata[indexdata['close'] != "收盘指数"]
indexdata = indexdata[indexdata['close'] != "没有单位"].reset_index(drop=True)
indexdata['date'] = indexdata['date'].astype(str).replace('\-', '', regex=True).astype(int)
# indexdata.date = pd.to_datetime(indexdata.date)
indexdata.to_hdf("indexdata.h5", "indexdata")

#%% 导入因子数据【已完成】
'''
data = pd.read_hdf("SSS.h5", "SSS")
# data = SSS[['Date', 'Stkcd', 'CLOSE', 'HIGH', 'LOW', 'OPEN', 'TURN', 'VWAP']]
data['Date'] = data['Date'].astype(str).replace('\-', '', regex=True).astype(int) # 日期格式
data.to_hdf("data.h5", "data")
'''
data = pd.read_hdf("data.h5", "data")

#%% 拆分保存【已完成】
# 导入
data = pd.read_hdf("data.h5", "data")

# 计算10天的回报 (t+1 to t+11)
data['return'] = data.groupby('Stkcd').apply(lambda x: (x.OPEN.shift(-11) / x.OPEN.shift(-1) - 1).fillna(0)).reset_index(level=0, drop=True)

# 
'''
data['return'] = data.groupby('Date').apply(
    lambda x: ((x['return'] - x['return'].mean()) / x['return'].std()).fillna(0)) \
    .reset_index(level=0, drop=True)
'''

# 数据图中的行数
# factor_list = ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'TURN', 'VWAP']
factor_list = ['Alpha2', 'Alpha13', 'Alpha14', 'Alpha15', 'Alpha18', 'Alpha20',
   'Alpha34', 'Alpha46', 'Alpha65', 'Alpha66', 'Alpha88', 'Alpha106',
   'Alpha167', 'Alpha3', 'Alpha189', 'Alpha19', 'Alpha38', 'Alpha31',
   'Alpha52', 'Alpha93', 'Alpha110', 'Alpha118', 'Alpha126', 'Alpha127',
   'Alpha129', 'Alpha153', 'Alpha171', 'Alpha183', 'Alpha187', 'Alpha22',
   'Alpha53', 'Alpha58', 'Alpha59', 'Alpha69', 'Alpha71', 'Alpha86',
   'Alpha161', 'Alpha165', 'Alpha6', 'Alpha8', 'Alpha10', 'Alpha12',
   'Alpha17', 'Alpha37', 'Alpha41', 'Alpha175', 'Alpha120', 'Alpha185',
   'Alpha78', 'Alpha166']

# 获得所有交易日
data = data.set_index(['Stkcd', 'Date'])
date_arr = data.index.get_level_values(1)
date_list = date_arr.unique()


for i, t in enumerate(tqdm(date_list)):
    # print("\n", i, t)
    
    # 自数据图像需要30天的历史后的前29天跳过
    if i < 29:
        continue

    # 选择最近30个交易日
    t_1 = date_list[i-29]
    tdata = data[(date_arr >= t_1) & (date_arr <= t)]

    # 选股
    stock_list = tdata.index.get_level_values(0).unique()

    # X: 所有股票数据的图片
    # Y: 所有股票未来10个交易日的收益，与X相同的顺序
    # asset: 所有股票代码，顺序与X相同
    X,Y,asset = [],[],[]

    for stock in stock_list:
        stock_t = tdata.loc[stock, :]
        if len(stock_t[factor_list].T.values[0]) == 30: # 防止数据不对齐
            X.append(stock_t[factor_list].T.values)
            Y.append(stock_t.loc[t, 'return'])
            asset.append(stock)

    np.save(r'pictures2/X_%d.npy' % t, np.array(X))
    np.save(r'pictures2/Y_%d.npy' % t, np.array(Y))
    np.save(r'pictures2/stock_%d.npy' % t, np.array(asset)) 

