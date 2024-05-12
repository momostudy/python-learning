#%% ===========================
#%% Part 2: 特征工程
#%% ===========================

#%
import numpy as np
import pandas as pd
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

#% 自定义函数
import numpy as np
import pandas as pd
import bottleneck as bn
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed


class CalculatorType:
    # 区分单变量和双变量
    SINGLE = 'Single_Variable'
    DUAL = 'Dual_Variable'

def parallel_cal_single_col(data: np.array, row: int, calculator):
    # 单变量并行运算
    feature = [calculator.cal(x[row]) for x in data]
    print('parallel_cal_single_col')
    return feature

def parallel_cal_dual_col(data: np.array, row1: int, row2: int, calculator):
    # 双变量并行运算
    feature = [calculator.cal(x[row1], x[row2]) for x in data]
    print('parallel_cal_dual_col')
    return feature

def cal_corr_cov(x, y, field='corr'):
    # 计算每一个x和y的cov或corr
    mu_x = x.mean(axis=1)
    mu_y = y.mean(axis=1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x和y的时间点数必须相同!')
    s_x = x.std(axis=1, ddof=n - 1)
    s_y = y.std(axis=1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                   mu_y[np.newaxis, :])
    
    # 这里要修改
    if False: # cov[0] == 0: # 
        return 0 # 
    else:
        if field == 'cov':
            return cov[0]
        elif field == 'corr':
            return (cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :]))[0]
        
        
def rolling_window(a: np.array, window_size: int, step: int):
    # 将1维数组转换为滚动窗口2维数组
    shape = ((a.shape[0] - window_size) // step + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class TSCorr:
    # 计算两个阵列之间的滚动关联
    type_ = CalculatorType.DUAL

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray, y: np.ndarray):
        X, Y = rolling_window(x, self.window, self.step), rolling_window(y, self.window, self.step)
        return cal_corr_cov(X, Y, "corr")

class TSCov:
    # 计算两个数组之间的滚动协方差
    type_ = CalculatorType.DUAL

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray, y: np.ndarray):
        X, Y = rolling_window(x, self.window, self.step), rolling_window(y, self.window, self.step)
        return cal_corr_cov(X, Y, "cov")


class TSStd:
    # 计算滚动标准差
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return bn.nanstd(X, 1)


class TSZscore:
    # 计算滚动z分数
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return bn.nanmean(X, 1) / bn.nanstd(X, 1)


class TSReturn:
    # 计算滚动回报率
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return X[:, -1] / X[:, 0] - 1


class TSDecaylinear:
    # 计算滚动衰减线性平均
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        weight = np.arange(self.window, 0, -1)
        weight = weight / weight.sum()
        return X @ weight


class TSMin:
    # 计算滚动最小值
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return bn.nanmin(X, 1)


class TSMax:
    # 计算滚动最大值
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return bn.nanmax(X, 1)


class TSMean:
    # 计算滚动均值
    type_ = CalculatorType.SINGLE

    def __init__(self, window: int, step: int):
        self.window = window
        self.step = step

    def cal(self, x: np.ndarray):
        X = rolling_window(x, self.window, self.step)
        return bn.nanmean(X, 1)

augmentation_map = {
    "ts_corr": TSCorr,
    "ts_cov": TSCov,
    "ts_stddev": TSStd,
    "ts_zscore": TSZscore,
    "ts_return": TSReturn,
    "ts_decaylinear": TSDecaylinear,
    "ts_min": TSMin,
    "ts_max": TSMax,
    "ts_mean": TSMean
}

#% 加载所有交易日
'''
data = pd.read_parquet("data.parquet")
date_list = data.date.unique()
'''
'''
data = pd.read_hdf("data.h5", "data")
data = data.set_index(['Stkcd', 'Date'])
date_arr = data.index.get_level_values(1)
date_list = date_arr.unique()
del data,date_arr

# import _pickle as pickle
# L0Close = pickle.load(open('L0Close.pkl','rb'))
import dill
dill.dump_session('date_list.pkl')
'''
import _pickle as pickle
pickle.load(open('date_list.pkl','rb')) # date_list

# 卷积层设置
config1 = {'calculator': ["ts_corr", "ts_cov", "ts_stddev",
                          "ts_zscore", "ts_return", "ts_decaylinear", "ts_mean"],
           "window": 10,
           "step": 10}

# 池化层设置
config2 = {'calculator': ["ts_min", "ts_max", "ts_mean"],
           "window": 3,
           "step": 3}

#% layer_treat
def layer_treat(data,config):
    results = []
    for calculator_name in tqdm(config['calculator']):
        calculator = augmentation_map[calculator_name](config['window'], config['step'])
        if calculator.type_ == CalculatorType.SINGLE: 
            results += Parallel(n_jobs=8)(delayed(parallel_cal_single_col)(data, row, calculator) for row in np.arange(data.shape[1]))
        elif calculator.type_ == CalculatorType.DUAL:
            results += Parallel(n_jobs=8)(delayed(parallel_cal_dual_col)(data, row1, row2, calculator) for row1, row2 in combinations(np.arange(data.shape[1]), 2))
        # print('results')
    df = np.hstack(results)
    df = df.reshape([df.shape[0], 1, df.shape[1]])    
    return df


#% 对每日进行运算
for i, t in enumerate(tqdm(date_list)):
    if i < 29:
        continue
    if t <= 20170515:
        continue
    
    # 读取当日数据图片
    X_t = np.load("pictures2/X_%d.npy" % t)
    
    # 卷积层
    data = layer_treat(X_t, config1)
    
    # 池化层
    data2 = layer_treat(data, config2)
    
    # 转变为二维序列，堆栈两个结果
    data = data.reshape([data.shape[0],data.shape[2]])
    data2 = data2.reshape([data2.shape[0], data2.shape[2]])
    feature = np.hstack([data,data2])

    # 保存文件
    np.save(r'features2/%d.npy' % t, feature)  
    
#%% END