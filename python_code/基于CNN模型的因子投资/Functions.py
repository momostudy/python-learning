#%% Functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import bottleneck as bn
import warnings
import datetime
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei', font_scale=1)

#%%
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pickle import dump

#%%
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#from main.feature import get_all_features
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import Sequential
from pickle import load
from sklearn.metrics import mean_squared_error

#%%
import sklearn #机器学习库
from sklearn.utils import shuffle #随机打乱数据集
from sklearn.model_selection import train_test_split #分割训练集

#%%
def stock_change(Set, StockCol):
    # 调整股票代码格式为整数 000001.SZ 变为 1
    Set[StockCol] = Set[StockCol].apply(lambda x: x[0:6]).astype(int)
    Set = Set.reset_index(drop=True)
    return Set

def date_change(Set, DateCol):
    # 将20000101变为2000-01-01
    Set[DateCol] = pd.to_datetime(Set[DateCol].astype(str), format='%Y%m%d')
    Set = Set.reset_index(drop=True)
    return Set

def date_change2(Set, DateCol):
    Set[DateCol] = pd.to_datetime(Set[DateCol])
    Set = Set.reset_index(drop=True)
    return Set

def get_return(Set, StockCol, DateCol, CloseCol):
    # 根据个股收盘价计算复利单日回报率
    Set.sort_values(by = [StockCol, DateCol], inplace=True)
    Set['RET'] = np.log(Set[CloseCol]/Set[CloseCol].shift(1))
    Set = Set.reset_index(drop=True)
    return Set[[StockCol, DateCol,'RET']].reset_index(drop=True)

def get_return2(Set, StockCol, DateCol, CloseCol):
    # 根据个股收盘价计算复利单日回报率
    Set.sort_values(by = [StockCol, DateCol], inplace=True)
    Set['RET'] = np.log(Set[CloseCol]/Set[CloseCol].shift(1))
    return Set.reset_index(drop=True)

def get_bench(Set, DateCol, IndexCol):
    # 抽出指定基准，并将指数换算为回报形式
    data = Set[[DateCol, IndexCol]]
    data['Stkcd'] = IndexCol
    data_ = get_return(data, 'Stkcd', DateCol, IndexCol)
    # data = data.rename(columns = {IndexCol:'RET'})
    return data_

def add_bench(Set, gg):
    # 合并指定基准
    Date_list = Set.drop_duplicates(
    subset = ['Date'], keep = 'last')
    Date_list = Date_list[['Date']]
    d = pd.merge(Date_list,gg.iloc[:,1:],on='Date',how='left')
    d['Stkcd'] = 'benchmark'
    df = pd.concat([Set,d])
    return df

def get_unique_list(Set, Col):
    return np.unique(Set[Col])

def get_date_id(Set, DateCol):
    Days = Set[DateCol].unique()
    ID_Days = list(range(1,len(Days)+1))
    T_Days = pd.DataFrame(columns=[DateCol,'ID_Day'],
                        index=np.arange(len(Days)))
    T_Days[DateCol] = Days
    T_Days['ID_Day'] = ID_Days
    Set = pd.merge(Set,T_Days,on=DateCol)
    return Set
    

#%%
def get_technical_indicators(data, StockCol, DateCol, CloseCol):
    # 添加技术因子
    
    data = data[[StockCol, DateCol, CloseCol]]
    
    # 创建7日和21日移动平均线
    data['MA7'] = data[CloseCol].rolling(window=7).mean()
    data['MA21'] = data[CloseCol].rolling(window=21).mean()

    # 创建MACD
    data['MACD'] = data[CloseCol].ewm(span=26).mean() - data[CloseCol].ewm(span=12,adjust=False).mean()

    # 创建布林带
    data['20SD'] = data[CloseCol].rolling(20).std()
    data['upper_band'] = data['MA21'] + (data['20SD'] * 2)
    data['lower_band'] = data['MA21'] - (data['20SD'] * 2)

    # 创建指数移动平均线
    data['EMA'] = data[CloseCol].ewm(com=0.5).mean()

    # 创建对数动量
    data['logmomentum'] = np.log(data[CloseCol] - 1)
    
    # 去除收盘价
    del data[CloseCol]
    data = data.reset_index(drop=True)

    return data

#%%
def get_fourier_transfer(dataset, StockCol, DateCol, CloseCol):
    # 得到傅里叶变换的列向量
    
    data_FT = dataset[[StockCol, DateCol, CloseCol]]

    close_fft = np.fft.fft(np.asarray(data_FT[CloseCol].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_com_df = pd.DataFrame()
    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        fft_ = np.fft.ifft(fft_list_m10)
        fft_com = pd.DataFrame({'fft': fft_})
        fft_com['absolute_of_' + str(num_) + '_comp'] = fft_com['fft'].apply(lambda x: np.abs(x))
        fft_com['angle_of_' + str(num_) + '_comp'] = fft_com['fft'].apply(lambda x: np.angle(x))
        fft_com = fft_com.drop(columns='fft')
        fft_com_df = pd.concat([fft_com_df, fft_com], axis=1)
    
    return fft_com_df

#%% S
def get_len(Set, Ym):
    Set = Set[Set['ym'] == Ym]
    return len(Set)


#%% S
def My_Resid(Set, Fac, Ym):
    
    Set = Set[Set['ym'] == Ym]
    Set = Set[['Stkcd', 'Date','Size', 'Size2', 'Dum_A',
       'Dum_B', 'Dum_C', 'Dum_D', 'Dum_E', 'Dum_F', 'Dum_G', 'Dum_H', 'Dum_I',
       'Dum_J', 'Dum_K', 'Dum_L', 'Dum_M', 'Dum_N', 'Dum_O', 'Dum_P', 'Dum_Q',
       'Dum_R', 'Dum_S',Fac]]
    
    Set = Set.fillna(0)
    formula_test = '%s~Size+Size2' % (Fac) +Inds_add
    result = sm.ols(formula=formula_test, data = FT_).fit()
    return list(Set['Stkcd']), list(Set['Date']), result.resid

#%%
def treat_X_label(Set, na_ratio = 0.3):
    # Set的格式为['Stkcd','Date', 其他因子]
    # 预处理：异常值、标准化
    # 剔除缺失率高于30%的因子特征，对部分缺失变量前向填充
    ret_fac = Set.loc[:,Set.count()>=Set.shape[0]*(1-na_ratio)]
    ret_fac.sort_values(by=['Stkcd','Date'],inplace=True)
    names = ret_fac.columns[2:-1].tolist()
    nn = len(names)
    for i in names:
        ret_fac[i] = ret_fac.groupby(['Stkcd'])[i].ffill()
        ret_fac[i] = ret_fac.groupby(['Date'])[i].apply(lambda x:x.fillna(x.mean()))
        ret_fac[i] = ret_fac.groupby(['Date'])[i].apply(lambda x:mad(x))
        ret_fac[i] = ret_fac.groupby(['Date'])[i].apply(lambda x:(x-x.mean())/x.std())
        print(str(nn),str(datetime.datetime.now()))
        nn = nn - 1
    return ret_fac

#%% 处理Y_label
def treat_Y_label(Set, date, updn_ratio = 0.2):
    # 标记训练集前20%上涨为1，后20%下跌为0
    Set = Set.fillna(0)
    train_set = Set[Set.ID_Day<date]
    train_set['ylabel'] = train_set.groupby(['ID_Day']
                          ).apply(lambda x:pd.qcut(x['RET'],
                             [0.,updn_ratio,1-updn_ratio,1.],
                              labels=False,duplicates = 'drop')).values
    train_set['ylabel'] = train_set['ylabel']/2
    train_set = train_set[train_set['ylabel']!=0.5]
    train_set['ylabel1'] = np.where(train_set['ylabel']==1,1,0)
    train_set['ylabel2'] = np.where(train_set['ylabel']==1,0,1)    
    
    # 测试集前50%为1，后50%为0
    test_set = Set[Set.ID_Day>=date]
    test_set['ylabel'] = test_set.groupby(['ID_Day']
                         )['RET'].apply(lambda x:pd.qcut(
                             x,2,labels=False,duplicates = 'drop')).values
    test_set['ylabel1'] = np.where(test_set['ylabel']==1,1,0)
    test_set['ylabel2'] = np.where(test_set['ylabel']==1,0,1)
    
    return train_set,test_set

#%% 特征工程
def FeatureSelect(ret_fac,train_set,test_set,threshold = 0.025):
    # 根据IC进行特征选择，筛选因子
    
    names = ret_fac.columns[2:-1].tolist()
    select = pd.DataFrame(
        train_set[names].corrwith(train_set['RET'],
                                  method='spearman')).dropna()
    slist = select[np.abs(select)>threshold].dropna().index.tolist()
    select[1] = np.sign(select[0])
    
    # 调整因子方向，测试集可以调整为正方向
    for i in slist:
        train_set[i] = train_set[i]*select.loc[i,1]
        test_set[i] = test_set[i]*select.loc[i,1]

    # 处理完毕，划分样本集与测试集，slist个输入，2个输出
    X_train = train_set[slist].values
    train_label = train_set[['ylabel1','ylabel2']].values
    X_test = test_set[slist].values
    test_label = test_set[['ylabel1','ylabel2']].values
    
    stock_list = ret_fac['Stkcd'].tolist()
    
    return X_train, train_label, X_test, test_label,stock_list

#%%
def save_file(Set, Set_name, n):
    file_name = Set_name + '_' + str(n) + '.txt'
    np.savetxt(file_name, Set, fmt="%d", delimiter=",") 
    
def load_file(Set_name, n):
    file_name = Set_name + '_' + str(n) + '.txt'
    return np.loadtxt(file_name, dtype=int, delimiter=",")
'''
# 保存二维数组
np.savetxt("matrix.txt", matrix, fmt="%d", delimiter=",")

# 读取二维数组
m = np.loadtxt("matrix.txt", dtype=int, delimiter=",")
'''

#%%
def dealwithNaN(data, method):
    # 空值填充处理
    if method == "zero":
        data = data.fillna(0)
    elif method == "mean":
        data = data.fillna(data.expanding().mean())
    elif method == "med":
        data = data.fillna(data.expanding().median())
    elif method == "pad":
        data = data.fillna(method='pad')  
    elif method == "ffill":
        data = data.fillna(method='ffill')
    elif method == "linear":
        data = data.interpolate(method='linear',
                                limit_direction='forward', axis=0)
    return data

#%%
def normal(a):
    min0 = np.min(a, axis=0)
    max0 = np.max(a, axis=0)
    return (a - min0) / (max0 - min0)

def min_max(Set):
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    return min_max_scaler.fit_transform(Set)

from sklearn.preprocessing import StandardScaler,Normalizer,Binarizer
def scaler(X, method):
    # 标准化、归一化、二值化
    if method == "Std":
       x = StandardScaler().fit(X) 
    elif method == "Nor":
       x = Normalizer().fit(X) 
    elif method == "Bin":  
       x = Binarizer(threshold=0.0).fit(X)
    return x.transform(X)    
    
#%%
def load_data(stock, look_back):
    # 创建训练集、测试集，给定股票数据和序列长度
    
    data_raw = stock
    data = []
    
    # 创建所有长度为seq_len的可能序列
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1,:]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

#%%
def get_X_y(X_data, y_data, n_steps_in = 3, n_steps_out = 1):
    # 用30天的数据来预测1天的价格
    # 得到X/y数据集
    X = list()
    y = list()
    yc = list()

    length = len(X_data)
    for i in range(0, length, 1):
        X_value = X_data[i: i + n_steps_in][:, :]
        y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
        yc_value = y_data[i: i + n_steps_in][:, :]
        if len(X_value) == 3 and len(y_value) == 1:
            X.append(X_value)
            y.append(y_value)
            yc.append(yc_value)

    return np.array(X), np.array(y), np.array(yc)


def predict_index(dataset, X_train, n_steps_in = 3, n_steps_out = 1):
    # 得到训练集预测指标

    # get the predict data (remove the in_steps days)
    train_predict_index = dataset.iloc[n_steps_in : X_train.shape[0] + n_steps_in + n_steps_out - 1, :].index
    test_predict_index = dataset.iloc[X_train.shape[0] + n_steps_in:, :].index

    return train_predict_index, test_predict_index


def split_train_test(data, rate = 0.7):
    # 分割训练、测试集
    train_size = round(len(data) * rate)
    data_train = data[0:train_size]
    data_test = data[train_size:]
    return data_train, data_test

#%% 分割
from datetime import timedelta 
def timerange_list(begin,end,N):
    # 给定起始日期和终止日期，输出起始日期按序排列的日期列表
    list1= []
    now = end
    for i in range(N-1):
        now = pd.to_datetime(now)
        start_ = str(datetime.datetime(now.year, now.month, 1))[0:10]
        end_ = str(pd.to_datetime(start_) - timedelta(days=1))[0:10]
        list1.append(start_)
        list1.append(end_)
        now = end_
    now = pd.to_datetime(now)
    list1.append(str(datetime.datetime(now.year, now.month, 1))[0:10])
    list1 = list1[::-1]
    list1.append(end[0:4]+"-"+end[4:6]+"-"+end[6:8])
    
    return list1


def get_s_e(l1:list):
    # 将日期列表分割成月初和月末列表
    l_start = []
    l_end = []
    for i in range(len(l1)):
        if i%2 == 0:
            l_start.append(l1[i])
        if i%2 == 1:
            l_end.append(l1[i])
            
    return l_start,l_end

'''示例调用
train_date_list = timerange_list("20171201","20181130",12)
train_start, train_end = get_s_e(train_date_list)

'''

#%%
def get_month_data(train_s,train_e,test_s,test_e):
    
    ending_day = str(list(get_price('000905.XSHG', start_date=train_s, end_date=train_e).index)[-1])[0:10]
    stock_list = list(get_index_weights("000905.XSHG", date= test_s).index)

    #factor panel: factors of Joinquant
    factor_data = get_factor_values(securities=stock_list, 
                                factors= factors,
                                start_date= ending_day , end_date=ending_day)
    df_factor = pd.DataFrame()
    factor_name = list(factor_data.keys())
    for name in factor_name:
        df_factor = pd.concat([df_factor,factor_data[name]])
    df_factor = df_factor.T
    df_factor.columns = factor_name
    #df_factor.isnull().sum()
    filter_list = list(df_factor.index)
    df_factor.index = filter_list
    
    
    #factor panel: factors of personally defined factors
    factor_data2 = calc_factors(filter_list, factors2, 
             start_date = ending_day, end_date = ending_day,  
             use_real_price=False, skip_paused=False)
    df_factor2 = pd.DataFrame()
    factor_name = list(factor_data2.keys())
    for name in factor_name:
        df_factor2 = pd.concat([df_factor2, factor_data2[name]])
    df_factor2 = df_factor2.T
    df_factor2.columns = factor_name
    df_factor2.index = filter_list
    
    #merge panel1 and panel2
    df_neu = pd.merge(df_factor, df_factor2, left_index = True, right_index = True)
    
    
    #data processing
    #df_neu = neutralize(df_factor, how=['sw_l1','market_cap'], date=ending_day, 
                    #axis=0, fillna='sw_l1', add_constant= True) 
    df_extreme = winsorize(df_neu, qrange=[0.05,0.93], 
                       inclusive=True, inf2nan=True, axis=0) #deal with extreme value
    df_stand = standardlize(df_extreme, inf2nan=True, axis=0) #standarlization


    #next month return
    return_next = []
    for code in filter_list:
        dq = np.array(get_price(code, start_date= test_s, end_date= test_e,
                 frequency='daily', skip_paused=False, fq='pre')['close'])
        return_next.append(dq[-1]/dq[0]-1)
    df_stand['return'] = return_next
    df_stand = df_stand[df_stand['return']!=0]

    #sort by return
    df_sort = df_stand.sort_values(by = "return", ascending=False).dropna(axis = 0)

    #pos and nega
    data_positive = df_sort.iloc[0:40,:]
    data_negative = df_sort.iloc[-40:,:]
    
    return data_positive, data_negative

def get_data(train_s:list,train_e:list,test_s:list,test_e:list):
    
    data_posi = pd.DataFrame()
    data_nega = pd.DataFrame()
    
    train_start = train_s
    train_end = train_e
    test_start = test_s
    test_end = test_e
    
    len_month = len(train_s)
    
    for j in range(len_month):
        
        train_s = train_start[j]
        train_e = train_end[j]
        test_s = test_start[j]
        test_e = test_end[j]
        
        df_pos,df_neg = get_month_data(train_s,train_e,test_s,test_e)
        data_posi = pd.concat([data_posi,df_pos])
        data_nega = pd.concat([data_nega,df_neg])
    
    
    data_posi['return'] = 1
    data_nega['return'] = 0
    data_ = pd.concat([data_posi,data_nega])
    data_ = shuffle(data_)
    
    return data_

#%% 模型评价
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true[:,3]-y_pred[:,3]))

def mae(y_true, y_pred):
    return tf.reduce_mean(tf.keras.backend.abs((y_true[:,3]-y_pred[:,3])))
    
def mape(y_true, y_pred):
    return tf.reduce_mean(tf.keras.backend.abs((y_true[:,3]-y_pred[:,3])/y_true[:,3]))
    
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:,3]-y_pred[:,3])))

def ar(y_true, y_pred):
    mask = tf.cast(y_pred[1:,3] > y_true[:-1,3],tf.float32)
    return tf.reduce_mean((y_true[1:,3]-y_true[:-1,3])*mask)

#%%
class Stand_TS_Generator(tf.keras.preprocessing.sequence.TimeseriesGenerator):
  def __getitem__(self, index):
    samples, targets  = super(Stand_TS_Generator, self).__getitem__(index)
    # shape : (n_batch, n_sequence, n_features)
    mean = samples.mean(axis=1)
    std = samples.std(axis=1)
    samples = (samples - mean[:,None,:])/std[:,None,:] #standarize along each feature
    # targets = (targets - mean[..., 3])/std[..., 3] # The close value is our target
    targets = (targets - mean)/std # The close value is our target
    return samples, targets

def get_gen_train_test(dataframe,n_sequence = 5,n_features = 7,n_batch = 32):
  data = dataframe.drop(columns='Date').to_numpy()
  #targets = data[:,3, None] 
  #add none to have same number of dimensions as data
  targets = data
  n_samples = data.shape[0]
  train_test_split=int(n_samples*0.9)

  data_gen_train = Stand_TS_Generator(data, targets,
                                length=n_sequence, sampling_rate=1,
                                stride=1, batch_size=n_batch,
                                start_index = 0,
                                end_index = train_test_split,
                                shuffle = True)
  data_gen_test = Stand_TS_Generator(data, targets,
                                length=n_sequence, sampling_rate=1,
                                stride=1, batch_size=n_batch,
                                start_index = train_test_split,
                                end_index = n_samples-1)

  return data_gen_train, data_gen_test

#%% 生成与判别
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def make_generator_model(n_sequence = 5,n_features = 7,n_batch = 32):

  inputs = Input(shape=(n_sequence, n_features,))
  lstm_1 = LSTM(units=10, return_sequences = True, activation=None, kernel_initializer='random_normal')(inputs)
  batch_norm1=tf.keras.layers.BatchNormalization()(lstm_1)
  lstm_1_LRelu = LeakyReLU(alpha=0.3)(batch_norm1) 
  lstm_1_droput = Dropout(0.3)(lstm_1_LRelu)
  lstm_2 = LSTM(units=10, return_sequences = False, activation=None, kernel_initializer='random_normal')(lstm_1_droput)
  batch_norm2=tf.keras.layers.BatchNormalization()(lstm_2)
  lstm_2_LRelu = LeakyReLU(alpha=0.3)(batch_norm2) 
  lstm_2_droput = Dropout(0.3)(lstm_2_LRelu)
  output_dense = Dense(n_features, activation=None)(lstm_2_droput)
  output = LeakyReLU(alpha=0.3)(output_dense) 
  model = Model(inputs = inputs, outputs = output)
  model.compile(loss=None, metrics = [mse , mae, mape, rmse, ar])
  model.summary()
  return model

def make_discriminator_model(n_sequence = 5,n_features = 7,n_batch = 32):
  model = Sequential()
  model.add(Flatten())
  model.add(Dense(units=72, input_shape=((n_sequence+1) * n_features,), activation=None, kernel_initializer='random_normal'))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  model.add(tf.keras.layers.GaussianNoise(stddev=0.2))
  model.add(Dropout(0.3))
  model.add(Dense(units=100, activation=None, kernel_initializer='random_normal'))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  model.add(Dropout(0.3))
  model.add(Dense(units=10, activation=None, kernel_initializer='random_normal'))
  model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  model.add(Dropout(0.3))
  model.add(Dense(1 ,activation='sigmoid'))
  model.compile(loss=discriminator_loss)
  return model

def make_generator_model2(input_dim, output_dim, feature_size) -> tf.keras.models.Model:
    model = Sequential()
    model.add(GRU(units=1024, return_sequences = True, 
                  input_shape=(input_dim, feature_size),
                  recurrent_dropout=0.2))
    model.add(GRU(units=512, return_sequences = True, # 256
                  recurrent_dropout=0.2)) 
    model.add(GRU(units=256, recurrent_dropout=0.2)) # 0.1
    model.add(Dense(128))
    model.add(Dense(64)) # 16
    model.add(Dense(units=output_dim))
    return model

def make_discriminator_model2():

    cnn_net = tf.keras.Sequential()
    cnn_net.add(Conv1D(32, input_shape=(4, 1), kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(64, kernel_size=5, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(128, kernel_size=5, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Flatten())
    cnn_net.add(Dense(220, use_bias=False))
    cnn_net.add(LeakyReLU())
    cnn_net.add(Dense(220, use_bias=False, activation='relu'))
    cnn_net.add(Dense(1, activation='sigmoid'))
    return cnn_net

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(x, y, fake_output):
    a1=0.01
    g_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)
    g_mse = tf.keras.losses.MSE(x, y)
    return a1*g_mse + (1-a1)*g_loss, g_mse

#%% 中位数极端值处理
def mad(series, n=3):
    median=series.quantile(0.5)
    diff_median=((series-median).abs()).quantile(0.5) #计算中位数标准差
    max_range=median+n*1.4826*diff_median
    min_range=median-n*1.4826*diff_median # 1.4826为调整系数
    return np.clip(series,min_range,max_range)

#%%
def transform_data(ret_fac,date,threshold = 0.025):
    # 因子的名称
    names = ret_fac.columns[2:-1].tolist()
    
    # 处理X_label
    # 预处理：异常值、标准化
    # 剔除缺失率高于30%的因子特征，对部分确实变量前向填充
    na_ratio = 0.3
    ret_fac = ret_fac.loc[:,ret_fac.count()>=ret_fac.shape[0]*(1-na_ratio)]
    miss = ret_fac.loc[:,ret_fac.count()<ret_fac.shape[0]].columns.tolist()
    ret_fac.sort_values(by=['Stkcd','Date'],inplace=True)
    for i in miss:
        ret_fac[i] = ret_fac.groupby(['Stkcd'])[i].ffill()
        ret_fac[i] = ret_fac.groupby(['Date'])[i].apply(lambda x:x.fillna(x.mean()))
    for i in names:
        ret_fac[i] = ret_fac.groupby(['Date'])[i].apply(lambda x:mad(x))
        ret_fac[i] = ret_fac.groupby(['Date'])[i].apply(lambda x:(x-x.mean())/x.std())
        
    # 处理Y_label
    # 标记训练集前20%上涨为1，后20%下跌为0
    updn_ratio = 0.2
    train_set = ret_fac[ret_fac.Date<date]
    train_set['ylabel'] = train_set.groupby(['Date']
                          )['RET'].apply(lambda x:pd.qcut(
                             [0.,updn_ratio,1-updn_ratio,1.],
                              labels=False,duplicates = 'drop'))
    train_set['ylabel'] = train_set['ylabel']/2
    train_set = train_set[train_set['ylabel']!=0.5]
    train_set['ylabel1'] = np.where(train_set['ylabel']==1,1,0)
    train_set['ylabel2'] = np.where(train_set['ylabel']==1,0,1)    
    
    # 测试集前50%为1，后50%为0
    test_set = ret_fac[ret_fac.Date>=date]
    test_set['ylabel'] = test_set.groupby(['Date']
                         )['RET'].apply(lambda x:pd.qcut(
                             x,2,labels=False,duplicates = 'drop'))
    test_set['ylabel1'] = np.where(test_set['ylabel']==1,1,0)
    test_set['ylabel2'] = np.where(test_set['ylabel']==1,0,1)
    
    # 根据IC进行特征选择，筛选因子
    select = pd.DataFrame(train_set[names].corrwith(train_set['RET'],
                                                    method='spearman')).dropna()
    slist = select[np.abs(select)>threshold].dropna().index.tolist()
    select[1] = np.sign(select[0])
    
    # 调整因子方向，测试集可以调整为正方向
    for i in slist:
        train_set[i] = train_set[i]*select.loc[i,1]
        test_set[i] = test_set[i]*select.loc[i,1]
    
    # 处理完毕，划分样本集与测试集，slist个输入，2个输出
    X_train = train_set[slist].values
    train_label = train_set[['ylabel1','ylabel2']].values
    X_test = test_set[slist].values
    test_label = test_set[['ylabel1','ylabel2']].values
    
    return X_train, train_label, X_test, test_label # , stock_list

#%%
def real_a(pred,real):
    # 识别前20%和后20%的预测能力
    compare = pd.DataFrame(np.hstack([pred,real]))
    compare2 = compare.sort_values(by=0,ascending=False).reset_index(drop=True)
    wins =[]
    for i in np.linspace(0.05,0.3,6):
        long = compare2.iloc[:int(compare2.shape[0]*i),[0,2]]
        winrate = long[2].sum()/long.shape[0]
        print(i,'多头正确率',long[2].sum()/long.shape[0])
        wins.append(winrate)
        short = compare2.iloc[int(compare2.shape[0](1-i)):,[0,3]]
        shortrate = short[3].sum()/short.shape[0]
        print(1-i,'空头正确率',shortrate)
    return wins
    
                  
#%%
def get_pred(ret_fac,end_month):
    # 输入13个月的面板，返回对应股票复合因子
    # 12：1有序划分训练集、测试集，训练集内按照9：1随机划分训练集与验证集
    X_train,train_label,X_test,test_label, stock_list = transform_data(ret_fac, date=end_month, threshold = 0.02) # 10:1预测
    print('训练集与测试集划分 Finish!')
    
    # 随机划分训练集与验证集，交叉验证备用，按9:1，这里仅按照一次分组
    np.random.seed()
    indices = np.random.permutation(X_train.shape[0])
    split = int(X_train.shape[0]*0.9)
    training_idx,test_idx = indices[:split],indices[split:]
    tran_tran,tran_vali = X_train[training_idx,:],X_train[test_idx,:]
    tran_label,vali_label = train_label[training_idx,:],train_label[test_idx,:]
    print('训练集与验证集划分 Finish!')
    
    # 滚动，每6个月训练一模型
    n = X_train.shape[1]
    slice_2 = int(n/2)+1
    slice_4 = int(n/4)+1
    layers = [[slice_2,slice_2],[slice_2,slice_4],[slice_4,slice_4]]
    batch_sizes = [250,500,750]
    dropouts = [0.5,0.75]
    
    evals = []
    evals2 = []
    portfolios = []
    preds = []
    for i in layers:
        for j in batch_sizes:
            for k in dropouts:
                start = time.time()
                portfolios.append((i,j,k))

                ann = ANN([n]+i+[2],tran_tran, tran_label, tran_vali, vali_label,j,k)
                acc = ann.session2()
                pred,real = ann.get_pred(tran_vali, vali_label)
                evals.append(real_a(pred,real))
                
                pred, real = ann.get_pred(X_test, test_label)
                evals2.append(real_a(pred,real)) # 按顺序保存
                preds.append(pred)
                end = time.time()
                print(portfolio[-1],acc,end-start) # 验证集准确率
    
    # 确定最优参数，取出相应模型的预测值，评判标准是多头预测胜率
    win_df = pd.DataFrame(evals, index = portfolios, columns = np.linspace(0.05,0.3,6))
    win_df['avg'] = np.mean(win_df.values, axis=1)
    win_df.reset_index(inplace=True)
    final_pred = preds[win_df['avg'].idxmax()]
    
    # 给出股票名+日期
    cross_section = pd.DataFrame(final_pred[:,0],
                                 columns = [end_month],
                                 index = stock_list).T # 当期预测上涨的概率
    
    return cross_section
    
#%%
def train_model(self,Loss,accuracy,train_op_Adam):
    start_timetime.time()
    losses = []
    its = []
    for epoch in range(seLf.nIter): 
        # 24个月,每次3个月限数据,每次迭代限数据；
        # 建议改为每个epoch可以迭代掉全部训练集,每次只打乱一次,batchs等份导入数据
        for batch_index in range(self.n_batches):
            X_batch,y_batch =  self.random_batch(self.X_train,self.train_Label,self.batch_size)
            self.sess.run(train_op_Adam,
                          feed_dict={self.x_tf:X_batch,
                                     self.y_tf:y_batch,
                                     self.keep_prob_s:self.keep_prob})
        if epoch % 100==99:
            Loss_val = self.sess.run([loss,accuracy]) ###
            test_acc = 0 ###
            elapsed = time.time()-start_time
            print("Epoch:",epoch,
                  "tLoss:",Loss-val,
                  "tAcc:",test_acc,
                  "time_cost",elapsed,)#样本内的训练误差
            start_timetime.time()
            Losses.append(loss_val)
            its.append(epoch)
    plt.scatter(x = np.array(its),y = np.array(losses))

#%% 回测系统
class Poet_Sim:
    # 回测系统
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
            i = interval[k]
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
        
        (rets[['l_ret','s_ret','l_minus_s']]+1
         ).cumprod().plot(title='多空组合净值曲线')
        (rets[['l_ret','benchmark','l_minus_b']]+1
         ).cumprod().plot(title='市场中性净值曲线')
        
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
            # 排序高于top的保留并计算均值
            LS = pd.DataFrame(
                np.nanmean(
                    np.where(x_qtl > top, Y, np.nan), axis=1
                ) - np.nanmean(
                    np.where(x_qtl < bottom, Y, np.nan),axis=1),
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
class GAN:
    def __init__(self, generator, discriminator, opt):
        self.opt = opt
        self.lr = opt["lr"]
        self.generator = generator
        self.discriminator = discriminator
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.batch_size = self.opt['bs']
        self.checkpoint_dir = '../training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, real_x, real_y, yc):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(real_x, training=True)
            generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
            d_fake_input = tf.concat([tf.cast(generated_data_reshape, tf.float64), yc], axis=1)
            real_y_reshape = tf.reshape(real_y, [real_y.shape[0], real_y.shape[1], 1])
            d_real_input = tf.concat([real_y_reshape, yc], axis=1)

            # Reshape for MLP
            # d_fake_input = tf.reshape(d_fake_input, [d_fake_input.shape[0], d_fake_input.shape[1]])
            # d_real_input = tf.reshape(d_real_input, [d_real_input.shape[0], d_real_input.shape[1]])

            real_output = self.discriminator(d_real_input, training=True)
            fake_output = self.discriminator(d_fake_input, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return real_y, generated_data, {'d_loss': disc_loss, 'g_loss': gen_loss}

    def train(self, real_x, real_y, yc, opt):
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_times'] = []
        train_hist['total_ptime'] = []

        epochs = opt["epoch"]
        for epoch in range(epochs):
            start = time.time()

            real_price, fake_price, loss = self.train_step(real_x, real_y, yc)

            G_losses = []
            D_losses = []

            Real_price = []
            Predicted_price = []

            D_losses.append(loss['d_loss'].numpy())
            G_losses.append(loss['g_loss'].numpy())

            Predicted_price.append(fake_price.numpy())
            Real_price.append(real_price.numpy())

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                tf.keras.models.save_model(generator, 'gen_model_3_1_%d.h5' % epoch)
                self.checkpoint.save(file_prefix=self.checkpoint_prefix + f'-{epoch}')
                print('epoch', epoch + 1, 'd_loss', loss['d_loss'].numpy(), 'g_loss', loss['g_loss'].numpy())
            # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            # For printing loss
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - start
            train_hist['D_losses'].append(D_losses)
            train_hist['G_losses'].append(G_losses)
            train_hist['per_epoch_times'].append(per_epoch_ptime)

        # Reshape the predicted result & real
        Predicted_price = np.array(Predicted_price)
        Predicted_price = Predicted_price.reshape(Predicted_price.shape[1], Predicted_price.shape[2])
        Real_price = np.array(Real_price)
        Real_price = Real_price.reshape(Real_price.shape[1], Real_price.shape[2])

        plt.plot(train_hist['D_losses'], label='D_loss')
        plt.plot(train_hist['G_losses'], label='G_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        return Predicted_price, Real_price, np.sqrt(mean_squared_error(Real_price, Predicted_price)) / np.mean(
            Real_price)

#%%

def FactorEvaluator2(factor, ret, quintile, LS=False, FE_output=False, dele=True):
    # 判断因子的收益预测能力（选股能力）
    # 算复合因子的预测能力
    
    # 输入factor 是矩阵形式的，行是日期，列是股票
    X = factor.values
    Y = ret.values
    if X.shape != Y.shape:
        print('X.shape:', X.shape, ';',
              'Y.shape:', Y.shape)
    
    # 算观测数
    N = len(X)
    
    # 返回给定形状和类型的用0填充的数组
    IC = np.zeros((N,))
    
    # 空值处理
    X = np.where(
        (~np.isnan(X) & (~np.isnan(Y))), X, np.nan)
    Y = np.where(
        (~np.isnan(X) & (~np.isnan(Y))), Y, np.nan)

    # 逐行做ranking，将非数字视为零 
    x_rk = bn.nanrankdata(X, axis=1)
    y_rk = bn.nanrankdata(Y, axis=1)   
    
    
    # portfolio sort
    if LS:
        x_qtl = x_rk / bn.nanmax(x_rk, axis=1)[:, np.newaxis]
        y_qtl = y_rk / bn.nanmax(y_rk, axis=1)[:, np.newaxis]
        
        bottom = 1.0 / quintile
        top = 1.0 - bottom
        
        LS = pd.DataFrame(
              np.nanmean(np.where(x_qtl > top, Y, np.nan), axis=1) 
            - np.nanmean(np.where(x_qtl < bottom, Y, np.nan),axis=1))
        
        # 返回给定轴上数组元素的累积总和，将非数字视为零      
        LS_cum = np.nancumsum(LS, axis=0)
        return LS_cum

    # 算IC，correlation coefficient
    for ii in range(N):
        IC[ii] = np.corrcoef(
            x_rk[ii][~np.isnan(x_rk[ii])],
            y_rk[ii][~np.isnan(y_rk[ii])])[0, 1]
        
    IC_a = np.nanmean(IC)
    IR_a = np.nanmean(IC) / np.nanstd(IC)


    if FE_output:
        print('Factor evaluation:')
        print('IC should be %f' % IC_a)
        print('IR should be %f' % IR_a)
        sns.set_style('whitegrid')
        plt.figure(figsize=(12, 8))
        plt.plot(IC, label='IC')
        plt.title("Information Correlation tendency")
        plt.show()

    return IC_a, IR_a


#%%
import numpy as np
import pandas as pd
import sqlalchemy as sa
import os
import configparser
import time
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


#%%

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_training_picture(data: pd.DataFrame, t: str, gap: int, num: int):
    """
    Generate training samples given the date
    All stocks, one picture every %gap% days for the last %num% trading days
    ---------------------
    :param data: pd.DataFrame, raw trading data
    :param t: str, date string like '2020-12-31'
    :param gap: int, data picture interval
    :param num: total number of days selected
    :return: np.array, 3d array containing all data pictures, should have
        a dimension of (#sample)*9*30
    ---------------------
    """
    factor_list = ['open', 'high', 'low', 'close', 'vwap', 'vol', 'pct', 'turnover', 'freeturn']
    stock_list = data.stockid.unique()
    X = []
    Y = []
    data = data.set_index(['stockid', 'date'])
    for stockid in tqdm(stock_list):
        stockdata = data.loc[(stockid, slice(None))]
        stock_picture = stockdata[stockdata.amount > 0][factor_list].T

        datearr = stock_picture.columns
        n = len(datearr[datearr <= t])
        if n <= 30:
            continue
        narr = np.arange(n - 12, max(30, n - 12 - gap*num), -gap)
        for i in narr:
            X.append(stock_picture.iloc[:, i - 30:i].values)
            Y.append(stockdata.loc[datearr[i], 'return'])
    return np.array(X), np.array(Y)


def generate_all_data_pictures(data: pd.DataFrame):
    """Generate data pictures as features and returns as labels for CNN model inputs
    ----------------------------
    :param data: pd.DataFrame
        All stock trading information in China A-Share Market
        From 2010.01.01 to 2020.12.31

    :return:
        No return
        Stock pictures are stored under /pictures
    ----------------------------
    """
    # calculate 10-days return, start from tomorrow (t+1 to t+11)
    data['return'] = data.groupby('stockid').apply(lambda x: (x.open.shift(-11) / x.open.shift(-1) - 1).fillna(0)) \
        .reset_index(level=0, drop=True)

    # standardize
    data['return'] = data.groupby('date').apply(
        lambda x: ((x['return'] - x['return'].mean()) / x['return'].std()).fillna(0)) \
        .reset_index(level=0, drop=True)

    # nine rows in data picture
    factor_list = ['open', 'high', 'low', 'close', 'vwap', 'vol', 'pct', 'turnover', 'freeturn']

    # get all trade days
    data = data.set_index(['stockid', 'date'])
    date_arr = data.index.get_level_values(1)
    date_list = date_arr.unique()

    for i, t in enumerate(tqdm(date_list)):
        # skip first 29 days since data picture needs 30-day history
        if i < 29:
            continue

        # select recent 30 days
        t_1 = date_list[i-29]
        tdata = data[(date_arr >= t_1) & (date_arr <= t)]

        # select stocks
        stock_list = tdata.index.get_level_values(0).unique()

        # X: all stock data pictures in a given day
        X = []
        # Y: all stock returns in a given day (returns for next 10 days), same order as X
        Y = []
        # asset: all stock ticker in a given day, same order as X
        asset = []

        for stock in stock_list:
            stock_t = tdata.loc[stock, :]
            if stock_t.shape[0] == 30 and stock_t.isnew.max() == 0 and \
                    stock_t.isst.max() == 0 and stock_t.istrade.max() == 1:
                X.append(stock_t[factor_list].T.values)
                Y.append(stock_t.loc[t, 'return'])
                asset.append(stock)

        # convert to np.array and stored as .npy files
        np.save(r'pictures/X_%s.npy' % t.strftime('%Y%m%d'), np.array(X))
        np.save(r'pictures/Y_%s.npy' % t.strftime('%Y%m%d'), np.array(Y))
        np.save(r'pictures/stock_%s.npy' % t.strftime('%Y%m%d'), np.array(asset))












