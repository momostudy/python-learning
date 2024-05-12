   
#%% ===========================
#%% Part 3: CNN模型
#%% ===========================

#%%
import numpy as np
import pandas as pd
import os
os.chdir('D:\FinTech_2022') 
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import bottleneck as bn
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei', font_scale=1)

#%% 
class CNNModel:
    """
    输入层：特征工程后的数据批归一化
    密集层：ReLU激活函数，50% dropout
    输出层：线性激活函数
    """

    def __init__(self, name, config, fit_config):
        self.name = name
        self.config = config
        self.fit_config = fit_config

        tf.random.set_seed(1)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.config['feat_num'],)),
            tf.keras.layers.BatchNormalization(),  # 数据归一化
            tf.keras.layers.Dense(units=30, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal()),
            tf.keras.layers.Dropout(rate=0.5),  # drop out 50%的神经元
            tf.keras.layers.Dense(units=1, activation='linear',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal())
        ])
        earlystop = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=10,
                                                     verbose=0, mode="min", baseline=None,
                                                     restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(config['model_path'], 
                                                        monitor='loss', verbose=0,
                                                        save_best_only=True, mode='min')
        self.cb_list = [earlystop, checkpoint]
        opt = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'], clipvalue=0.5)
        self.model.compile(loss=self.config['loss'], optimizer=opt, metrics=self.config['metrics'])

    def fit(self, x, y):
        self.model.fit(
            x=x,
            y=y,
            callbacks=self.cb_list,
            **self.fit_config
        )

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred




#%% 单步调试
# cnn_result(20180102, 20220613)
begt, endt = 20180102, 20220613
config = {
    'model_path': Path('models'),
    'feat_num': 702,
    'learning_rate': 0.002,
    'loss': 'mse',
    'metrics': 'mse',
}

fit_config = {
    'batch_size': 2000,
    'epochs': 10000,
}
# read data
# data = pd.read_parquet("data.parquet")
# data = pd.read_hdf("data.h5", "data") # 
# get all trade days
# date_list = data.date.unique()
import _pickle as pickle
pickle.load(open('date_list.pkl','rb')) # date_list
beg_n = np.argmax(date_list[date_list <= begt]) + 1 # 
end_n = np.argmax(date_list[date_list <= endt]) # 

#%% 数据查看
t = 20180102
# X_t = np.load("features2/%d.npy" % t)
Y_t = np.load("pictures2/Y_%d.npy" % t)
X_t = np.load("pictures2/X_%d.npy" % t)
stock_t = np.load("pictures2/stock_%d.npy" % t)

# CNN_2019-01-01_2020-12-31.parquet
e = pd.read_parquet("D:/FinTech_2022/CNN_2019-01-01_2020-12-31.parquet")
e = e.reset_index()
np.save(r'e.npy', e) 


#%% 单步测试
begt, endt = 20180102, 20220613
df_predict = pd.DataFrame(columns=['date', 'stock', 'score'])
import _pickle as pickle
pickle.load(open('date_list.pkl','rb')) # date_list
beg_n = np.argmax(date_list[date_list <= begt]) + 1 # 
end_n = np.argmax(date_list[date_list <= endt]) # 

#%%
for n in tqdm(np.arange(beg_n, end_n, 1)):
    x_train_raw = []
    y_train = []
    for i in np.arange(n, n - 30, -5):
        t = date_list[i]
        X_t = np.load("pictures2/X_%d.npy" % t)# 
        Y_t = np.load("pictures2/Y_%d.npy" % t)# 
        if len(x_train_raw):
            x_train_raw = np.concatenate((x_train_raw, X_t), axis=0)
        else:
            x_train_raw = X_t
        if len(y_train):
            y_train = np.concatenate((y_train, Y_t), axis=0)
        else:
            y_train = Y_t        

    # 打乱数据顺序
    shuffle = np.random.permutation(x_train_raw.shape[0])
    x_train_raw = x_train_raw[shuffle]
    y_train = y_train[shuffle]
 
    predict_t = date_list[n - 1] #
    stock_predict = np.load("pictures2/stock_%d.npy" % predict_t)#
    y_predict = y_train[:len(stock_predict)]
      
    score_t = pd.DataFrame(columns=['stock', 'score'])
    score_t['stock'] = stock_predict
    score_t['score'] = y_predict
    
    score_t.score = score_t.score.astype(float)
    score_t['date'] = date_list[n+5] # 
    score_t = score_t.dropna()

    df_predict = pd.concat([df_predict, score_t])
    # print(date_list[n])

#%% 保存预测结果 
# df_predict = df_predict.set_index(['date', 'stock'])
# df_predict = df_predict.dropna()
df_predict = df_predict[df_predict['score']<=100]
print(df_predict)

df_predict['rank'] = df_predict.groupby(['date'])['score'].rank(ascending = False)
# print(df_predict)

df_predict_100 = df_predict[df_predict['rank']<=100].reset_index(drop=True)
df_predict_100 = df_predict_100.sort_values(by = ["date", "rank"]).reset_index(drop=True)
del df_predict_100['rank']
print(df_predict_100)
df_predict.to_csv('df_predict_3.csv', encoding='utf_8_sig') 
df_predict_100.to_csv('df_predict_3_100.csv', encoding='utf_8_sig') 
     

#%% 预测结果 -
df_predict = pd.DataFrame(columns=['date', 'stock', 'score'])
for n in tqdm(np.arange(beg_n, end_n, 10)):
    # 基线模型的训练输入，从12天前开始。
    # 对于交易日t，只能使用t-1数据进行预测，然后进行训练 
    # 数据应从t-12开始，返回标签从t-11到t-1
    x_train_raw = []
    y_train = []
    for i in np.arange(n - 12, n - 512, -5):
        t = date_list[i] # pd.to_datetime(date_list[i]).strftime('%Y%m%d') # 这里格式需要调整为整数！
        X_t = np.load("features2/%d.npy" % t)# 
        Y_t = np.load("pictures2/Y_%d.npy" % t)# 
        if len(x_train_raw):
            x_train_raw = np.concatenate((x_train_raw, X_t), axis=0)
        else:
            x_train_raw = X_t
        if len(y_train):
            y_train = np.concatenate((y_train, Y_t), axis=0)
        else:
            y_train = Y_t

    # 删除 nan/inf 数据
    isnum = ~ (np.isnan(x_train_raw).max(axis=1) | np.isinf(x_train_raw).max(axis=1))
    x_train_raw = x_train_raw[isnum]
    y_train = y_train[isnum]

    # 打乱数据顺序
    shuffle = np.random.permutation(x_train_raw.shape[0])
    x_train_raw = x_train_raw[shuffle]
    y_train = y_train[shuffle]

    # 模型训练，调用CNNModel类 
    selector = CNNModel('CNN_selector_on_%d' % t, config, fit_config) # 
    selector.fit(x_train_raw, y_train)

    # 预测今日标签
    predict_t = date_list[n - 1] # pd.to_datetime(date_list[n - 1]).strftime('%Y%m%d')  # 这里格式需要调整为整数！
    x_predict = np.load("features2/%d.npy" % predict_t)#
    stock_predict = np.load("pictures2/stock_%d.npy" % predict_t)#
    y_predict = selector.predict(np.asarray(x_predict).astype(np.float32))
    
    # 与股票ID合并
    score_t = pd.DataFrame(np.transpose([stock_predict, y_predict.reshape(-1)]), columns=['stock', 'score'])
    
    # 转换数据类型并去掉空值预测
    score_t.score = score_t.score.astype(float)
    score_t['date'] = date_list[n] # pd.to_datetime(date_list[n])
    score_t = score_t.dropna()
    
    # 保存每期得分
    score_t.to_hdf("predictions/score_t_%d.h5" % n)
    print("score_t_%d saved!" % n)

    df_predict = pd.concat([df_predict, score_t])
    print(date_list[n])

# 保存预测结果
df_predict = df_predict.set_index(['date', 'stock'])
# indexdata.to_hdf("indexdata.h5", "indexdata")
# df_predict.to_parquet("predictions/CNN_%d_%d.parquet" % (begt, endt))
df_predict.to_hdf("predictions/CNN_%d_%d.h5" % (begt, endt))



#%% ==============
#%% 初步回测
#%% ==============

#%% 整理因子文件
# df_predict = pd.read_csv('df_predict.csv')
FT2 = df_predict[['score', 'stock', 'date']].drop_duplicates(subset = ['stock', 'date'], keep = 'last')
FT2.columns=['score', 'Stkcd','Date']
FT2['Stkcd'] = FT2['Stkcd'].astype(int)
FT2['Date'] = FT2['Date'].astype(int).astype(str)
FT2['Date'] = FT2['Date'].apply(lambda x: x[:4]+"-"+x[4:6]+"-"+x[6:8])
FT2 = FT2.reset_index(drop=True)
LL = FT2['Date'].unique() # 日期列表

# 得到因子文件
FT2 = FT2.replace([np.inf, -np.inf], np.nan)
FT2 = FT2.fillna(0)
df3 = FT2.set_index(['Stkcd','Date'])
factor_file = df3.score.unstack(0)

factor_file.replace([np.inf, -np.inf], np.nan)
factor_file = factor_file.fillna(0)

from sklearn.preprocessing import MinMaxScaler
values = factor_file.values # 将dataframe转化为array
values = values.astype('float32') # 定义数据类型
tool = MinMaxScaler(feature_range=(0, 1))
data = tool.fit_transform(values)
ff = pd.DataFrame(data) 

ff.index = factor_file.index
ff.columns = factor_file.columns

ff.to_csv('factor_file_hyk_0626_ff.csv',
                   encoding='utf_8_sig') 

#%% 导入沪深300收益率数据
def date_change2(Set, DateCol):
    Set[DateCol] = pd.to_datetime(Set[DateCol])
    Set = Set.reset_index(drop=True)
    return Set

bench_1 = pd.read_excel('IDX_Idxtrd_1.xlsx')
bench_2 = pd.read_excel('IDX_Idxtrd_2.xlsx')
bench = pd.concat([bench_1.iloc[2:,1:],bench_2.iloc[2:,1:]])
bench = bench.rename(columns = {
    'Idxtrd01':'Date',
    'Idxtrd08':'RET'})
bench = date_change2(bench, 'Date')

bench.to_hdf("bench.h5", "bench")
bench['Stkcd'] = 'benchmark'

#%% 导入回报率数据
os.chdir('D:\FinTech_2022') 
SSS = pd.read_hdf("SSS.h5", "SSS")
SSS = SSS[['Stkcd','Date','RET']]

df = pd.concat([SSS,bench])
df["ymd"] = df["Date"].astype(str).replace('\-', '', regex=True).astype(int)
df = df[df['ymd']>=20180101]

#%% 得到回报率文件
df_ = df[['Stkcd','Date','RET']].drop_duplicates(subset = ['Stkcd','Date'], keep = 'last')

'''
LL_ = pd.DataFrame(LL)
LL_.columns=['Date']
LL_['Date'] = pd.to_datetime(LL_['Date'])

# 调整与因子文件同频？
df = pd.merge(df,LL_,on='Date')
'''

ret_file = df_.set_index(['Stkcd','Date']).RET.unstack(0)
# ret_file  = ret_file.iloc[:-1,:]
ret_file.to_csv('ret_file_hyk_0626.csv',
                   encoding='utf_8_sig') 

#%% 回测结果

#%% 单步测试
rolling_ret,rolling_factors,direct = ret_file.iloc[6:-1],ff.iloc[1:],-1
benchmark = rolling_ret['benchmark']
rolling_ret = rolling_ret.drop(columns='benchmark')
if direct==-1:
    rolling_factors = -rolling_factors

#%%
decile=0.1
long = []
short = []
rolling_factors = rolling_factors.shift(1)
interval = rolling_ret.index[1:]
for k in tqdm(range(len(interval)),desc="Get return: ",ncols=100):
    i = str(interval[k])[:10] # 修改
    fac = rolling_factors.loc[i,:].dropna() # 根据前一天收盘时的因子决策
    short_list = fac[fac<=fac.quantile(decile)].index.tolist() # 做空后10%
    long_list = fac[fac>=fac.quantile(1-decile)].index.tolist() # 做多前10%
    
    lret = rolling_ret.loc[i,long_list].fillna(0).mean()
    sret = rolling_ret.loc[i,short_list].fillna(0).mean()
    
    long.append(lret)
    short.append(sret)

rets = pd.DataFrame([long,short],index=['l_ret','s_ret'],columns = rolling_ret.index[1:]).T
rets = pd.concat([rets,benchmark],axis=1).dropna()
rets['l_minus_b'] = rets['l_ret']-rets['benchmark']
rets['l_minus_s'] = rets['l_ret']-rets['s_ret']

(rets[['l_ret','s_ret','l_minus_s']]+1).cumprod().plot(title='多空组合净值曲线')
(rets[['l_ret','benchmark','l_minus_b']]+1).cumprod().plot(title='市场中性净值曲线')

#%% 
nav = rets['l_ret'] 
rf_annual=0

nav = pd.DataFrame(nav.copy())
nav.columns=['ret']
# 默认第一天为净值为1
nav['nav'] = (nav['ret']+1).cumprod()
nav['drawdown'] = nav['nav']/nav['nav'].cummax()-1
annal_ret = nav.iloc[-1,1]**(252/nav.shape[0])-1

nav['down_ret'] = np.where(nav['ret']>0,0,nav['ret'])
semi_std = np.sqrt((nav['down_ret']**2).mean()) * np.sqrt(252)

# 计算月收益
nav['month'] = nav.index
nav['month'] = nav['month'].apply(lambda x:str(x)[:7])
nav2 = nav.drop_duplicates(subset=['month'],keep='last')
first_row = pd.DataFrame([1.]*len(nav2.columns),index=nav2.columns).T
nav2 = first_row.append(nav2)
nav2['mret'] = nav2['nav'].pct_change()
nav2.dropna(inplace=True)

# 所有指标均不带百分比
dicts = {'年化收益率':annal_ret,
    '年化波动率':nav['ret'].std()*252**(1/2),
    '区间最大回撤':nav['drawdown'].min(),
    '夏普比率':(annal_ret-rf_annual)/(nav['ret'].std()*252**(1/2)),
    '卡玛比率':(annal_ret-rf_annual)/np.abs(nav['drawdown'].min()),
    '索提诺比率':(annal_ret-rf_annual)/semi_std,
    '单月最大跌幅':nav2.mret.min(),
    '区间策略胜率':nav[nav['ret']>0].shape[0]/nav.shape[0],
    '区间策略赔率':1-nav[nav['ret']>0].shape[0]/nav.shape[0],
    'One day-5% VaR':-nav['ret'].quantile(0.05)}

a1 = pd.DataFrame(dicts,index=[0]).T
print(a1)
a1.to_csv('总体情况.csv',encoding='utf_8_sig') 

#%%
def get_p(nav):
    nav = pd.DataFrame(nav.copy())
    nav.columns=['ret']
    # 默认第一天为净值为1
    nav['nav'] = (nav['ret']+1).cumprod()
    nav['drawdown'] = nav['nav']/nav['nav'].cummax()-1
    annal_ret = nav.iloc[-1,1]**(252/nav.shape[0])-1
    
    nav['down_ret'] = np.where(nav['ret']>0,0,nav['ret'])
    semi_std = np.sqrt((nav['down_ret']**2).mean()) * np.sqrt(252)
    
    # 计算月收益
    nav['month'] = nav.index
    nav['month'] = nav['month'].apply(lambda x:str(x)[:7])
    nav2 = nav.drop_duplicates(subset=['month'],keep='last')
    first_row = pd.DataFrame([1.]*len(nav2.columns),index=nav2.columns).T
    nav2 = first_row.append(nav2)
    nav2['mret'] = nav2['nav'].pct_change()
    nav2.dropna(inplace=True)
    
    # 所有指标均不带百分比
    dicts = {'年化收益率':annal_ret,
        '年化波动率':nav['ret'].std()*252**(1/2),
        '区间最大回撤':nav['drawdown'].min(),
        '夏普比率':(annal_ret-rf_annual)/(nav['ret'].std()*252**(1/2)),
        '卡玛比率':(annal_ret-rf_annual)/np.abs(nav['drawdown'].min()),
        '索提诺比率':(annal_ret-rf_annual)/semi_std,
        '单月最大跌幅':nav2.mret.min(),
        '区间策略胜率':nav[nav['ret']>0].shape[0]/nav.shape[0],
        '区间策略赔率':1-nav[nav['ret']>0].shape[0]/nav.shape[0],
        'One day-5% VaR':-nav['ret'].quantile(0.05)}
    return pd.DataFrame(dicts,index=[0]).T    
    
    
# 分年度返回策略评价指标
ret_name = 'l_minus_b'
ret_year = rets.reset_index()
ret_year['year'] = ret_year['Date'].apply(lambda x:str(x)[:4])
perform_year = pd.DataFrame()
for i in ret_year.year.unique():
    temp = ret_year[ret_year.year==i]
    res = get_p(temp[ret_name])
    perform_year = pd.concat([perform_year,res],axis=1)
perform_year.columns = ret_year.year.unique()
aa = pd.DataFrame(np.array(perform_year).real)
aa.columns = perform_year.columns
aa.index = perform_year.index
print(aa)
aa.to_csv('分年度情况.csv',encoding='utf_8_sig') 

#%%
class Poet_Sim:
    
    def __init__(self,rolling_ret,rolling_factors,direct=1):
        self.rolling_ret = rolling_ret.drop(columns='benchmark')
        self.rolling_factors = rolling_factors # 滞后一期决定持仓比例确定当日收益
        self.benchmark = rolling_ret['benchmark']
        self.direct = direct
        if self.direct==-1:
            self.rolling_factors = -self.rolling_factors
        
    def get_strategyret(self,decile=0.1):
        long = []
        short = []
        rolling_factors = self.rolling_factors.shift(1)
        # for i in self.rolling_ret.index[1:]:
        interval = self.rolling_ret.index[1:]
        for k in tqdm(range(len(interval)),desc="Get return: ",ncols=100):
            i = interval[k][:10] # 修改这里
            fac = rolling_factors.loc[i,:].dropna() # 根据前一天收盘时的因子决策
            short_list = fac[fac<=fac.quantile(decile)].index.tolist() # 做空后10%
            long_list = fac[fac>=fac.quantile(1-decile)].index.tolist() # 做多前10%
            
            lret = self.rolling_ret.loc[i,long_list].fillna(0).mean()
            sret = self.rolling_ret.loc[i,short_list].fillna(0).mean()
            
            long.append(lret)
            short.append(sret)
        
        rets = pd.DataFrame([long,short],index=['l_ret','s_ret'],columns = self.rolling_ret.index[1:]).T
        rets = pd.concat([rets,self.benchmark],axis=1).dropna()
        rets['l_minus_b'] = rets['l_ret']-rets['benchmark']
        rets['l_minus_s'] = rets['l_ret']-rets['s_ret']
        
        (rets[['l_ret','s_ret','l_minus_s']]+1).cumprod().plot(title='多空组合净值曲线')
        (rets[['l_ret','benchmark','l_minus_b']]+1).cumprod().plot(title='市场中性净值曲线')
        
        return rets
        
    def get_performance(self,nav:pd.Series,rf_annual=0.02,show=True):
        # 计算IR时采用rf=0;
        nav = pd.DataFrame(nav.copy())
        nav.columns=['ret']
        # 默认第一天为净值为1
        nav['nav'] = (nav['ret']+1).cumprod()
        nav['drawdown'] = nav['nav']/nav['nav'].cummax()-1
        annal_ret = nav.iloc[-1,1]**(252/nav.shape[0])-1
        
        nav['down_ret'] = np.where(nav['ret']>0,0,nav['ret'])
        semi_std = np.sqrt((nav['down_ret']**2).mean()) * np.sqrt(252)
        
        # 计算月收益
        nav['month'] = nav.index
        nav['month'] = nav['month'].apply(lambda x:str(x)[:7])
        nav2 = nav.drop_duplicates(subset=['month'],keep='last')
        first_row = pd.DataFrame([1.]*len(nav2.columns),index=nav2.columns).T
        nav2 = first_row.append(nav2)
        nav2['mret'] = nav2['nav'].pct_change()
        nav2.dropna(inplace=True)
        
        # 所有指标均不带百分比
        dicts = {'年化收益率':annal_ret,
            '年化波动率':nav['ret'].std()*252**(1/2),
            '区间最大回撤':nav['drawdown'].min(),
            '夏普比率':(annal_ret-rf_annual)/(nav['ret'].std()*252**(1/2)),
            '卡玛比率':(annal_ret-rf_annual)/np.abs(nav['drawdown'].min()),
            '索提诺比率':(annal_ret-rf_annual)/semi_std,
            '单月最大跌幅':nav2.mret.min(),
            '区间策略胜率':nav[nav['ret']>0].shape[0]/nav.shape[0],
            '区间策略赔率':-nav[nav['ret']>0].ret.mean()/nav[nav['ret']<=0].ret.mean(),
            'One day-5% VaR':-nav['ret'].quantile(0.05)}
        if show:
            for k,y in dicts.items():
                print(k,format(y, '.3f'))
            print('\n注：输入l_minus_b序列以及设定无风险利率为0, 年化夏普 = 信息比率.')
        
        # #最大回撤修复时间
        # lowest = nav['drawdown'].idxmin()
        # revert_time = nav[(nav['drawdown']==0)&(nav.index>lowest)]
        # print('最大回撤点发生在',lowest)
        # if revert_time.empty:
        #     print('截至目前最大回撤未修复.')
        # else:
        #     print('\n最大回撤修复时间',nav.loc[lowest:revert_time.index[0],:].shape[0]-1,'天.')
        
        return pd.DataFrame(dicts,index=[0]).T
    
    def get_cost_tur(self,fee1=0.001,fee2=0.0005):
        # 多头:考虑单边千分之一印花税,双边万分之五券商佣金;
        long_nav = []
        long_list = []
        daily_tur = []
        costs = []
        llong_list = pd.DataFrame()
        # for i in self.rolling_ret.index:
        for k in tqdm(range(self.rolling_ret.shape[0]),desc="Get return: ",ncols=100):
            i = self.rolling_ret.index[k]
            # 确定备选股票
            fac = self.rolling_factors.loc[i,:].dropna() 
            if i == self.rolling_ret.index[0]:
                # 第一天
                long_list_ = fac[fac>=fac.quantile(0.9)].index.tolist() # 做多前10%
                tur = 1
                cost = fee2
                nav = 1-fee2
            else:
                # 第N天,N>1
                llong_list = long_list.rename(columns={'w':'lw'})
                long_list_ = fac[fac>=fac.quantile(0.9)].index.tolist() # 做多前10%
                
                # 获取昨持仓收益率
                ret_list = self.rolling_ret.loc[i,llong_list.index.tolist()].fillna(0)
                ret_list.name = 'ret'
                compare = pd.concat([ret_list,llong_list],axis=1)
                compare['before_w'] = (compare['ret']+1) * compare['lw'] # 获取调仓前净值
                
                # 获取今持仓净值
                long_list = pd.DataFrame(long_list_,columns=['stock']).set_index('stock')
                long_list['after_w'] = compare.before_w.sum()/long_list.shape[0]
                
                # 计算买卖金额
                compare2 = pd.concat([compare,long_list],axis=1).fillna(0)
                adj_pos = compare2['after_w']-compare2['before_w']
                
                # 计算券商佣金以及印花税,换手率
                tur = np.abs(adj_pos).sum() / compare.before_w.sum()
                cost = adj_pos[adj_pos>0].sum()*fee2 - adj_pos[adj_pos<0].sum()*(fee1+fee2)
                nav = compare.before_w.sum() - cost
                cost = cost / compare.before_w.sum()
            
            long_nav.append(nav) # long_nav记录真实净值,w记录真实仓位
            costs.append(cost)
            daily_tur.append(tur)
            long_list = pd.DataFrame(long_list_,columns=['stock'])
            long_list['w'] = long_nav[-1]/long_list.shape[0] # 更新当前仓位
            long_list.set_index('stock',inplace=True)
        
        long_nav_acost = pd.DataFrame([long_nav,costs,daily_tur],columns=self.rolling_ret.index,index=['nav','cost','turnover']).T
        long_nav_acost['rets_c'] = long_nav_acost['nav'].pct_change()
        rets = pd.concat([long_nav_acost,self.benchmark],axis=1).dropna()
        rets['rets_c_b'] = rets['rets_c'] - rets['benchmark']
        
        # 返回平均日换手率,以及扣除费用后的PNL,收益率序列
        long_nav_acost['nav'].plot()
        print('平均每日双边换手率为',np.round(np.mean(daily_tur),5))
        print('平均每日费率为',np.round(np.mean(costs),5))
        return rets

    def FactorEvaluator(self, quintile, LS=False, FE_output=True):
        X = self.rolling_factors.shift(1).values # 滞后一期
        Y = self.rolling_ret.values
        if X.shape != Y.shape:
            print('X.shape:', X.shape, ';', 'Y.shape:', Y.shape)
        # 只要收益率或者因子特征有nan,都设定为nan
        X, Y = np.where((~np.isnan(X) & (~np.isnan(Y))), X, np.nan), np.where((~np.isnan(X) & (~np.isnan(Y))), Y, np.nan)
        # 对np.array排序,在指定轴升序排序
        x_rk, y_rk = bn.nanrankdata(X, axis=1), bn.nanrankdata(Y, axis=1)    
        # 逐行获得有效个数并计算排序百分比
        x_qtl = x_rk / bn.nanmax(x_rk, axis=1)[:, np.newaxis]

        if LS: # 快速获得Long-short序列
            bottom = 1.0 / quintile
            top = 1.0 - bottom
            # 排序高于top的保留并计算均值,np.mean不能跳过nan,但是np.nanmean可以
            LS = pd.DataFrame(
                np.nanmean(np.where(x_qtl > top, Y, np.nan), axis=1) - np.nanmean(np.where(x_qtl < bottom, Y, np.nan),
                                                                              axis=1),
                columns = ['l_minus_s'],index=self.rolling_factors.index.tolist())
            # LS_cum = np.nancumsum(LS, axis=0)
            # return LS_cum
            return LS
        
        # 获取10分位数据
        LS_10 = pd.DataFrame()
        deciles = np.linspace(0,1,quintile+1)
        for i in range(len(deciles)-1):
            L = pd.DataFrame(np.nanmean(np.where((x_qtl <= deciles[i+1])&(x_qtl > deciles[i]), Y, np.nan), axis=1))
            LS_10 = pd.concat([LS_10,L],axis=1)        
        LS_10.columns = ['decile '+str(i) for i in range(1,11)]
        LS_10.index = self.rolling_factors.index.tolist()
        # 作图
        sns.set_context('paper', rc={'lines.linewidth':.75})
        (LS_10.fillna(0)+1).cumprod().plot(title='Deciles of Factor Net Value')
        
        # 获取每一行return与factor的排序的相关系数
        N = len(X)
        IC = np.zeros((N,))
        for ii in range(N):
            IC[ii] = np.corrcoef(x_rk[ii][~np.isnan(x_rk[ii])], y_rk[ii][~np.isnan(y_rk[ii])])[0, 1]
        IC_a = np.nanmean(IC)
        IR_a = np.nanmean(IC) / np.nanstd(IC)
        print('IC: ', IC_a, 'IR: ',IR_a)
        IC = pd.DataFrame(IC,index=self.rolling_factors.index.tolist(),columns=['IC'])
        
        # 输出每月Rank_IC
        if FE_output:
            IC['date'] = IC.index
            IC['month'] = IC['date'].apply(lambda x:str(x)[:7])
            IC_m = IC.groupby(['month'])['IC'].mean().reset_index()
            IC_m.set_index(['month'],inplace=True)
            IC_m.plot(title = "Monthly Information Correlation Tendency")
        
        return pd.concat([LS_10,IC['IC']],axis=1)

#%%
# ret_file = ret_file.fillna(0)
# factor_file = factor_file.fillna(0)

# 输出收益率,因子特征,方向
poet_sim = Poet_Sim(ret_file.iloc[6:-1],ff.iloc[1:],direct=-1) 

# 0.1表示十分位,0.2表示五分位
rets = poet_sim.get_strategyret(decile=0.1) 

# 每次只能输入一个收益率Series
results = poet_sim.get_performance(rets['l_ret'],
                                   rf_annual=0,
                                   show=False) 

# 第一个是单边印花税,第二个是双边券商佣金
rets_c = poet_sim.get_cost_tur(fee1=0.001,fee2=0.0005) 

# 分年度返回策略评价指标
ret_name = 'l_minus_b'
ret_year = rets.reset_index()
ret_year['year'] = ret_year['Date'].apply(lambda x:str(x)[:4])
perform_year = pd.DataFrame()
for i in ret_year.year.unique():
    temp = ret_year[ret_year.year==i]
    res = poet_sim.get_performance(temp[ret_name],
                                   rf_annual=0,
                                   show=False)
    perform_year = pd.concat([perform_year,res],axis=1)
perform_year.columns = ret_year.year.unique()
print(perform_year)

























