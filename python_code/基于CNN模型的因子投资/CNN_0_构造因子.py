#%% FinalMain

#%% 
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
os.chdir('D:\FinTech_2022') 
import _pickle as pickle
import time
import tensorflow as tf
import datetime

# 导入自定义函数包
from Functions import *

#%% 导入数据
'''
closeprice_adjaf.pkl
highestprice_adjaf.pkl
lowestprice_adjaf.pkl
openprice_adjaf.pkl
turnoverrate.pkl
vwap.pkl
'''
namespace = globals()
load_list = ['closeprice_adjaf',
             'highestprice_adjaf',
             'lowestprice_adjaf',
             'openprice_adjaf',
             'turnoverrate',
             'vwap']
for i in load_list:
    namespace['%s' % (i)] = pickle.load(open(i+'.pkl','rb'))

#%% 变换形态
for i in load_list:
    Set = namespace['%s' % (i)]
    Set_ = Set.stack().reset_index()
    Set_ = Set_.rename(columns = {0:i})
    namespace['%s_' % (i)] = Set_

#%% 拼接数据
SSS = namespace['%s_' % (load_list[0])]
for i in load_list[1:]:
    Set_ = namespace['%s_' % (i)]
    SSS = pd.concat([SSS,Set_[i]],axis = 1)
    
print(SSS)

#%% 修改名称
SSS = SSS.rename(columns = {
    'closeprice_adjaf':'CLOSE',
    'highestprice_adjaf':'HIGH',
    'lowestprice_adjaf':'LOW',
    'openprice_adjaf':'OPEN',
    'turnoverrate':'TURN',
    'vwap':'VWAP',
    'tradeDate':'Date',
    'secID':'Stkcd'})

SSS = stock_change(SSS,'Stkcd')
SSS = date_change2(SSS,'Date')
SSS = get_return2(SSS,'Stkcd','Date','CLOSE') # 得到 RET

#%% 构建函数

# SUM(A, n) 序列A 过去n 天求和
def SUM(Set, Col, n):
    l = Set.groupby('Stkcd'
                )[Col].rolling(n).sum().reset_index(drop = True)
    return l

# MEAN(A, n) 序列A 过去n 天均值
# Ex: MEAN(SSS, 'RET', 60)
def MEAN(Set, Col, n):
    l = Set.groupby('Stkcd'
                )[Col].rolling(n).mean().reset_index(drop = True)
    return l

# STD(A, n) 序列A 过去n 天标准差
# Ex: STD(SSS, 'RET', 60)
def STD(Set, Col, n):
    l = Set.groupby('Stkcd'
                )[Col].rolling(n).std().reset_index(drop = True)
    return l

# TSMIN(A, n) 序列A 过去n 天的最小值
# Ex: TSMIN(SSS, 'RET', 60)
def TSMIN(Set, Col, n):
    l = Set.groupby('Stkcd'
                )[Col].rolling(n).min().reset_index(drop = True)
    return l

# TSMAX(A, n) 序列A 过去n 天的最大值
# Ex: TSMAX(SSS, 'RET', 60)
def TSMAX(Set, Col, n):
    l = Set.groupby('Stkcd'
                )[Col].rolling(n).max().reset_index(drop = True)
    return l

# PROD(A, n) 序列A 过去n 天累乘

# SUMAC(A, n) 计算A 的前n 项的累加
# Ex: SUMAC(SSS, 'RET', 60)
def SUMAC(Set, Col, n):
    l = Set.groupby('Stkcd'
                )[Col].cumsum().reset_index(drop = True)
    return l



# CORR(A, B, n) 序列A、B 过去n 天相关系数
# Ex: CORR(SSS, 'HIGH', 'LOW', 60) 
def CORR_(x,n,a,b):
    return pd.DataFrame(x[a].rolling(n).corr(x[b]))
def CORR(Set, A, B, n):
    return Set.groupby('Stkcd')[[A, B]].apply(CORR_,n,A,B)

# COVIANCE (A, B, n)
# Ex: COVIANCE(SSS, 'HIGH', 'LOW', 60) 
def COVIANCE_(x,n,a,b):
    return pd.DataFrame(x[a].rolling(n).cov(x[b]))
def COVIANCE(Set, A, B, n):
    return Set.groupby('Stkcd')[[A, B]].apply(COVIANCE_,n,A,B)

# RANK(A) 向量A 升序排序(无用)
# Ex: RANK(SSS, 'Date')
def RANK(Set, Col):
    return Set.sort_values(by=['Stkcd',Col])[Col]

# SIGN(A) 符号函数
# Ex: SIGN(SSS, 'HIGH')
def SIGN(Set, Col):
    l = np.where(Set[Col] > 0,1,
                 np.where(Set[Col] == 0,0,-1))
    return l

# DELAY(A, n)
# Ex: DELAY(SSS, 'HIGH', 2)
def DELAY(Set, Col, n):
    l = Set.groupby(['Stkcd']).shift(n)[Col]
    return l

# DELTA(A, n) 
# Ex: DELTA(SSS, 'HIGH', 2)
def DELTA(Set, Col, n):
    l = Set[Col] - Set.groupby(['Stkcd']).shift(n)[Col]
    return l

# TSRANK(A, n) 序列A 的末位值在过去n 天的顺序排位

# HIGHDAY(A, n) 计算A 前n 期时间序列中最大值距离当前时点的间隔

# LOWDAY(A, n) 计算A 前n 期时间序列中最大值距离当前时点的间隔

# COUNT(condition, n) 计算前n 期满足条件condition 的样本个数

# WHERE A？B:C 若A 成立，则为B，否则为C
def WHERE(Cond, T, F):
    l = np.where(Cond,T,F)
    return l

# MAX(A, B) 在A,B 中选择最大的数
# Ex: MAX(SSS, 'OPEN', 'CLOSE')
def MAX(Set, Col1, Col2):
    l = Set[[Col1, Col2]].max(axis=1)
    return l    

# MIN(A, B) 在A,B 中选择最小的数
def MIN(Set, Col1, Col2):
    l = Set[[Col1, Col2]].min(axis=1)
    return l 

# REGBETA(A, B, n) 前n 期样本A 对B 做回归所得回归系数
# 形如REGBETA(CLOSE,SEQUENCE(6))
def REGBETA(Set, Col, n):
    return 0

# REGRESI(A, B, n) 前n 期样本A 对B 做回归所得的残差



#%% 构建因子50个

#%% Alpha2
SSS['a'] = (((SSS['CLOSE']-SSS['LOW'])-(SSS['HIGH']-SSS['CLOSE']))/(
                                     SSS['HIGH']-SSS['LOW']))
SSS['Alpha2'] = -1*DELTA(SSS,'a',1)
del SSS['a']

#%% Alpha3

# 
SSS['L'] = DELAY(SSS,'CLOSE',1)
# MIN(LOW,DELAY(CLOSE,1))
SSS['T'] = MIN(SSS,'LOW','L')
# MAX(HIGH,DELAY(CLOSE,1))
SSS['F'] = MAX(SSS,'LOW','L')
# (CLOSE>DELAY(CLOSE,1)? T:F
SSS['W'] = WHERE( SSS['CLOSE']>DELAY(SSS,'CLOSE',1), SSS['T'], SSS['F'])
# (CLOSE=DELAY(CLOSE,1)?0:CLOSE - W
SSS['X'] = WHERE( SSS['CLOSE']==DELAY(SSS,'CLOSE',1), 0, SSS['CLOSE']) - SSS['W']

SSS['Alpha3'] = SUM(SSS, 'X', 6)
del SSS['L'],SSS['T'],SSS['F'],SSS['W'],SSS['X'] 

#%% Alpha6
# (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
SSS['a'] = (SSS['OPEN'] * 0.85) + (SSS['HIGH'] * 0.15)
SSS['b'] = DELAY(SSS,'a',4)
SSS['Alpha6'] = SIGN(SSS,'b')*(-1)
del SSS['a'],SSS['b']

#%% Alpha8
# RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
SSS['a'] = (SSS['HIGH']+SSS['LOW'])/ 2 * 0.2 + (SSS['VWAP'] * 0.8)
SSS['Alpha8'] = DELTA(SSS,'a',4)*(-1)
del SSS['a']

#%% Alpha10
'''
RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
'''
SSS['a'] = WHERE(SSS['RET']<0,
                 STD(SSS,'RET',20),
                 SSS['CLOSE'])**2
SSS['Alpha10'] = TSMAX(SSS,'a',5)
del SSS['a']

#%% Alpha12
# ((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (ABS((CLOSE - VWAP))))
SSS['Alpha12'] = (SSS['OPEN'] - SUM(SSS,'VWAP',10)/10) * -1 * abs(SSS['CLOSE']-SSS['VWAP'])


#%% Alpha13
SSS['Alpha13'] = (((SSS['HIGH'] * SSS['LOW'] )**0.5) - SSS['VWAP'])

#%% Alpha14
SSS['Alpha14'] = SSS['CLOSE'] - DELAY(SSS,'CLOSE',5)

#%% Alpha15
SSS['Alpha15'] = SSS['OPEN']/DELAY(SSS,'CLOSE',1)-1

#%% Alpha17
# (VWAP - MAX(VWAP, 15))^DELTA(CLOSE, 5)
SSS['Alpha17'] = (SSS['VWAP'] - TSMAX(SSS,'VWAP',15))**DELTA(SSS,'CLOSE', 5)

#%% Alpha18
SSS['Alpha18'] = SSS['CLOSE'] / DELAY(SSS,'CLOSE',5)

#%% Alpha19
'''
(
 CLOSE<DELAY(CLOSE,5)?
 (CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):
     (CLOSE=DELAY(CLOSE,5)?
      0:
      (CLOSE-DELAY(CLOSE,5))/CLOSE)
)
'''
SSS['a'] = WHERE(
                 SSS['CLOSE']==DELAY(SSS,'CLOSE',5),
                 0,(SSS['CLOSE']-DELAY(SSS,'CLOSE',5))/SSS['CLOSE'])
SSS['Alpha19'] = WHERE(
                 SSS['CLOSE']<DELAY(SSS,'CLOSE',5),
                 (SSS['CLOSE']-DELAY(SSS,'CLOSE',5))/DELAY(SSS,'CLOSE',5),SSS['a'])
del SSS['a']

#%% Alpha20
SSS['Alpha20'] = (SSS['CLOSE']-DELAY(SSS,'CLOSE',6))/DELAY(SSS,'CLOSE',6)*100 

#%% Alpha22
'''
MEAN(
     (CLOSE-MEAN(CLOSE,6))
     /MEAN(CLOSE,6)
     -DELAY(
          (CLOSE-MEAN(CLOSE,6))
          /MEAN(CLOSE,6)
          ,3),
     12)
'''
SSS['a'] = (SSS['CLOSE']-MEAN(SSS,'CLOSE',6))/MEAN(SSS,'CLOSE',6)
SSS['b'] = SSS['a'] - DELAY(SSS,'a',3)
SSS['Alpha22'] = MEAN(SSS,'b',12)
del SSS['a'],SSS['b']

#%% Alpha26-too big
# ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))
'''
SSS['a'] = DELAY(SSS,'CLOSE',5)
SSS['Alpha26'] = SUM(SSS,'CLOSE',7) / 7 - SSS['CLOSE'] + CORR(SSS,'a','VWAP',230)
del SSS['a']
'''
#%% Alpha31
# (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
SSS['Alpha31'] = (SSS['CLOSE']-MEAN(SSS,'CLOSE',12))/MEAN(SSS,'CLOSE',12)*100

#%% Alpha34
SSS['Alpha34'] = MEAN(SSS,'CLOSE',12)/SSS['CLOSE']

#%% Alpha37
# (-1 * ((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10)))
SSS['a'] = SUM(SSS,'OPEN', 5) * SUM(SSS,'RET', 5)
SSS['Alpha37'] = -1 * (SSS['a'] - DELAY(SSS,'a',10))
del SSS['a']

#%% Alpha38
# SUM(HIGH, 20) / 20 < HIGH ? (-1 * DELTA(HIGH, 2)) : 0							
SSS['Alpha38'] = WHERE(
                       SUM(SSS,'HIGH', 20) / 20 < SSS['HIGH'],
                       -1 * DELTA(SSS,'HIGH', 2),0)

#%% Alpha41
# MAX(DELTA((VWAP), 3), 5))* -1
SSS['a'] = DELTA(SSS,'VWAP', 3)
SSS['Alpha41'] = TSMAX(SSS,'a',5)*-1
del SSS['a']

#%% Alpha46
SSS['Alpha46'] = (MEAN(SSS,'CLOSE',3) + MEAN(SSS,'CLOSE',6) + MEAN(SSS,'CLOSE',12) + MEAN(SSS,'CLOSE',24))/(4*SSS['CLOSE'])

#%% Alpha52
'''
SUM(
    MAX(0,
        HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)
        ),26)/
SUM(
    MAX(0,
        DELAY((HIGH+LOW+CLOSE)/3,1)
        -LOW),26)
*100
'''
SSS['a'] = (DELAY(SSS,'HIGH',1)+DELAY(SSS,'LOW',1)+DELAY(SSS,'CLOSE',1))/3
SSS['S1'] = WHERE(SSS['HIGH']-SSS['a']>0,SSS['HIGH']-SSS['a'],0)
SSS['S2'] = WHERE(SSS['a']>0,SSS['a'],0)-SSS['LOW']
SSS['Alpha52'] = SUM(SSS,'S1',26)/SUM(SSS,'S2',26)*100
del SSS['a'],SSS['S1'],SSS['S2']

#%% Alpha53
'''
COUNT(CLOSE>DELAY(CLOSE,1),12)
/12*100
'''
SSS['a'] = SSS['CLOSE'] - DELAY(SSS,'CLOSE',1)
SSS['b'] = WHERE(SSS['a']>0,1,0)
SSS['Alpha53'] = SUMAC(SSS, 'b', 12)/12*100
del SSS['a'],SSS['b']

#%% Alpha54-too big
# (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
'''
SSS['a'] = abs(SSS['CLOSE'] - SSS['OPEN'])
SSS['Alpha54'] = -1*(STD(SSS,'a',10)+SSS['CLOSE'] - SSS['OPEN']+CORR(SSS,'CLOSE','OPEN',10))
del SSS['a']
'''

#%% Alpha58
# COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
SSS['a'] = SSS['CLOSE'] - DELAY(SSS,'CLOSE',1)
SSS['b'] = WHERE(SSS['a']>0,1,0)
SSS['Alpha58'] = SUMAC(SSS, 'b', 20)/20*100
del SSS['a'],SSS['b']

#%% Alpha59
'''
SUM(
    (
     CLOSE=DELAY(CLOSE,1)?
     0:
     CLOSE-(CLOSE>DELAY(CLOSE,1)?
            MIN(LOW,DELAY(CLOSE,1)):
                MAX(HIGH,DELAY(CLOSE,1)))
         )
    ,20)
'''
SSS['T'] = WHERE(SSS['LOW']>DELAY(SSS,'CLOSE',1),DELAY(SSS,'CLOSE',1),SSS['LOW'])
SSS['F'] = WHERE(SSS['HIGH']>DELAY(SSS,'CLOSE',1),SSS['HIGH'],DELAY(SSS,'CLOSE',1))
SSS['C'] = WHERE(SSS['CLOSE']>DELAY(SSS,'CLOSE',1),SSS['T'],SSS['F'])
SSS['a'] = WHERE(SSS['CLOSE']==DELAY(SSS,'CLOSE',1),0,SSS['C'])
SSS['Alpha59'] = SUM(SSS,'a',20)
del SSS['a'],SSS['T'],SSS['F'],SSS['C']


#%% Alpha65
SSS['Alpha65'] = MEAN(SSS,'CLOSE',6) / SSS['CLOSE']

#%% Alpha66
SSS['Alpha66'] = (SSS['CLOSE']-MEAN(SSS,'CLOSE',6))/MEAN(SSS,'CLOSE',6)*100

#%% Alpha69
'''
SUM(DTM,20)>SUM(DBM,20)?
 (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):
    (
     SUM(DTM,20)=SUM(DBM,20)?
     0:
         (SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)
         )
'''
# DTM (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
# DBM (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
SSS['DTM'] = WHERE(SSS['OPEN']<=DELAY(SSS,'OPEN',1),0,
                   WHERE(SSS['HIGH']-SSS['OPEN']>SSS['OPEN']-DELAY(SSS,'OPEN',1),
                         SSS['HIGH']-SSS['OPEN'],SSS['OPEN']-DELAY(SSS,'OPEN',1)))
SSS['DBM'] = WHERE(SSS['OPEN']<=DELAY(SSS,'OPEN',1),0,
                   WHERE(SSS['OPEN']-SSS['LOW']>SSS['OPEN']-DELAY(SSS,'OPEN',1),
                         SSS['OPEN']-SSS['LOW'],SSS['OPEN']-DELAY(SSS,'OPEN',1)))

SSS['Alpha69'] =  WHERE(SUM(SSS,'DTM',20)>SUM(SSS,'DBM',20),
                        (SUM(SSS,'DTM',20)-SUM(SSS,'DBM',20))/SUM(SSS,'DTM',20),
                        WHERE(SUM(SSS,'DTM',20)==SUM(SSS,'DBM',20),0,
                              (SUM(SSS,'DTM',20)-SUM(SSS,'DBM',20))/SUM(SSS,'DBM',20)))
del SSS['DTM'],SSS['DBM']

#%% Alpha71
SSS['Alpha71'] = (SSS['CLOSE']-MEAN(SSS,'CLOSE',24))/MEAN(SSS,'CLOSE',24)*100

#%% Alpha78
# ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
SSS['a'] = abs(SSS['CLOSE']-(MEAN(SSS,'HIGH',12)+MEAN(SSS,'LOW',12)+MEAN(SSS,'CLOSE',12))/3)
SSS['Alpha78'] = ((SSS['HIGH']+SSS['LOW']+SSS['CLOSE'])/3-(MEAN(SSS,'HIGH',12)+MEAN(SSS,'LOW',12)+MEAN(SSS,'CLOSE',12))/3
                  )/0.015/MEAN(SSS,'a',12)
del SSS['a']
    
#%% Alpha86
'''
(0.25  <  (((DELAY(CLOSE,  20)  -  DELAY(CLOSE,  10))  /  10)  -  ((DELAY(CLOSE,  10)  -  CLOSE)  /  10)))  ? 
 (-1  *  1)  :
(((((DELAY(CLOSE,  20)  -  DELAY(CLOSE,  10))  /  10)  -  ((DELAY(CLOSE,  10)  -  CLOSE)  /  10))  <  0)  ?
  1  : 
      ((-1  *  1)  * (CLOSE - DELAY(CLOSE, 1))))
'''
SSS['Alpha86'] = WHERE(
    0.25  <  (((DELAY(SSS,'CLOSE',  20)  -  DELAY(SSS,'CLOSE',  10))  /  10)  -  ((DELAY(SSS,'CLOSE',  10)  -  SSS['CLOSE'])  /  10)),-1,
    WHERE(
        ((DELAY(SSS,'CLOSE',  20)  -  DELAY(SSS,'CLOSE',  10))  /  10  -  (DELAY(SSS,'CLOSE',  10)  -  SSS['CLOSE'])  /  10)  <  0
        ,1,-1* (SSS['CLOSE'] - DELAY(SSS,'CLOSE', 1))
                       ))
                              



#%% Alpha88
SSS['Alpha88'] = (
    SSS['CLOSE']-DELAY(SSS,'CLOSE',20)
    )/DELAY(SSS,'CLOSE',20)*100

#%% Alpha93
'''
SUM(
    (OPEN>=DELAY(OPEN,1)
     ?0:
         MAX((OPEN-LOW),
             (OPEN-DELAY(OPEN,1)))),
    20)

'''
SSS['a'] = SSS['OPEN']-SSS['LOW']
SSS['b'] = SSS['OPEN']-DELAY(SSS,'OPEN',1)
SSS['c'] = WHERE(SSS['OPEN']>=DELAY(SSS,'OPEN',1),0,MAX(SSS,'a','b'))
SSS['Alpha93'] = SUM(SSS,'c',20)
del SSS['a'],SSS['b'],SSS['c']

#%% Alpha106
SSS['Alpha106'] = SSS['CLOSE'] / DELAY(SSS,'CLOSE',20)

#%% Alpha110
'''
SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)
/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)
*100
'''
SSS['a'] = DELAY(SSS,'LOW',1)
SSS['S1'] = WHERE(SSS['HIGH']-SSS['a']>0,SSS['HIGH']-SSS['a'],0)
SSS['S2'] = WHERE(SSS['a']-SSS['LOW']>0,SSS['a']-SSS['LOW'],0)
SSS['Alpha110'] = SUM(SSS,'S1',20)/SUM(SSS,'S2',20)*100
del SSS['a'],SSS['S1'],SSS['S2']

#%% Alpha118
'''
SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
'''
SSS['S1'] = SSS['HIGH']-SSS['OPEN']
SSS['S2'] = SSS['OPEN']-SSS['LOW']
SSS['Alpha118'] = SUM(SSS,'S1',20)/SUM(SSS,'S2',20)*100
del SSS['S1'],SSS['S2']

#%% Alpha120
# (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
SSS['Alpha120'] = (SSS['VWAP'] - SSS['CLOSE']) / (SSS['VWAP'] + SSS['CLOSE'])


#%% Alpha126
# (CLOSE+HIGH+LOW)/3
SSS['Alpha126'] = (SSS['CLOSE']+SSS['HIGH']+SSS['LOW'])/3

#%% Alpha127
'''
(
 MEAN(
      (
       100*(CLOSE-MAX(CLOSE,12))/MAX(CLOSE,12)
                )^2
     )
 )^(1/2)
'''
SSS['a'] = WHERE(SSS['CLOSE']>12,SSS['CLOSE'],12)
SSS['b'] = 100*((SSS['CLOSE']-SSS['a'])/SSS['a'])**2
SSS['Alpha127'] = MEAN(SSS,'b',12)**0.5
del SSS['a'],SSS['b']

#%% Alpha129
'''
SUM(
    (CLOSE-DELAY(CLOSE,1)<0?
     ABS(CLOSE-DELAY(CLOSE,1)):
         0)
    ,12)
'''
SSS['a'] = SSS['CLOSE'] - DELAY(SSS,'CLOSE',1)
SSS['b'] = WHERE(SSS['a']<0,abs(SSS['a']),0)
SSS['Alpha129'] = SUM(SSS,'b',12)
del SSS['a'],SSS['b']

#%% Alpha153
# (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
SSS['Alpha153'] = (MEAN(SSS,'CLOSE',3)+MEAN(SSS,'CLOSE',6)+MEAN(SSS,'CLOSE',12)+MEAN(SSS,'CLOSE',24))/4

#%% Alpha161
'''
MEAN(
     MAX(
         MAX(
             HIGH-LOW,
             ABS(DELAY(CLOSE,1)-HIGH)),
         ABS(DELAY(CLOSE,1)-LOW)
         )
     ,12)
'''
SSS['a'] = WHERE(SSS['HIGH']-SSS['LOW']>abs(DELAY(SSS,'CLOSE',1)-SSS['HIGH']),
                 SSS['HIGH']-SSS['LOW'],abs(DELAY(SSS,'CLOSE',1)-SSS['HIGH']))
SSS['b'] = WHERE(SSS['a']>abs(DELAY(SSS,'CLOSE',1)-SSS['LOW']),
                 SSS['a'],abs(DELAY(SSS,'CLOSE',1)-SSS['LOW']))
SSS['Alpha161'] = MEAN(SSS,'b',12)
del SSS['a'],SSS['b']                      

#%% Alpha165
# MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
SSS['a'] = SSS['CLOSE']-MEAN(SSS,'CLOSE',48)
SSS['b'] = SUMAC(SSS,'CLOSE',48)
SSS['Alpha165'] = TSMAX(SSS,'a',48)-TSMIN(SSS,'b',48)/STD(SSS,'CLOSE',48)
del SSS['a'],SSS['b']

#%% Alpha166
'''
-20*(20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
'''
SSS['a'] = SSS['CLOSE']/DELAY(SSS,'CLOSE',1)
SSS['Alpha166'] = -20*(20-1)**1.5*SUM(SSS,'a',20)-1-MEAN(SSS,'a',20)/((20-1)*(20-2)*(SUM(SSS,'a',20)**2))**1.5
del SSS['a']


#%% Alpha167
'''
Alpha167 SUM(
    (CLOSE-DELAY(CLOSE,1)>0?
     CLOSE-DELAY(CLOSE,1):
         0),
    12)
'''
SSS['a'] = WHERE(
                 SSS['CLOSE']-DELAY(SSS,'CLOSE',1)>0,
                 DELAY(SSS,'CLOSE',1),0)

SSS['Alpha167'] = SUM(SSS,'a',12)
del SSS['a']

#%% Alpha171
# ((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))
SSS['Alpha171'] = -1*(SSS['LOW']-SSS['CLOSE'])*SSS['OPEN']**5 / (
    (SSS['CLOSE']-SSS['HIGH'])*SSS['CLOSE']**5)

#%% Alpha175
# MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
SSS['a'] = WHERE(SSS['HIGH']-SSS['LOW']>abs(DELAY(SSS,'CLOSE',1)-SSS['HIGH']),
                 SSS['HIGH']-SSS['LOW'],abs(DELAY(SSS,'CLOSE',1)-SSS['HIGH']))
SSS['b'] = WHERE(SSS['a']>abs(DELAY(SSS,'CLOSE',1)-SSS['LOW']),
                 SSS['a'],abs(DELAY(SSS,'CLOSE',1)-SSS['LOW']))
SSS['Alpha175'] = MEAN(SSS,'b',6)
del SSS['a'],SSS['b']   

#%% Alpha183
'''
MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))
-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
'''
SSS['a'] = SSS['CLOSE']-MEAN(SSS,'CLOSE',24)
SSS['Alpha183'] = max(SSS['a']) - min(SSS['a'])/STD(SSS,'CLOSE',24)
del SSS['a']

#%% Alpha185
# RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
SSS['Alpha185'] = -1*(1-(SSS['OPEN']/SSS['CLOSE'])**2)

#%% Alpha187
'''
SUM(
    (OPEN<=DELAY(OPEN,1)?
     0:
         MAX(
             (HIGH-OPEN),(OPEN-DELAY(OPEN,1))
             ))
    ,20)
'''
SSS['a'] = WHERE(SSS['OPEN']<=DELAY(SSS,'OPEN',1),0,
                 WHERE((SSS['HIGH']-SSS['OPEN'])>(SSS['OPEN']-DELAY(SSS,'OPEN',1)),
                       SSS['HIGH']-SSS['OPEN'],SSS['OPEN']-DELAY(SSS,'OPEN',1))
                     )
SSS['Alpha187'] = SUM(SSS, 'a', 20)
del SSS['a']


#%% Alpha189
# Alpha189	MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
SSS['a'] = abs(SSS['CLOSE']-DELAY(SSS,'CLOSE',6))
SSS['Alpha189'] = MEAN(SSS,'a',6)
del SSS['a']

#%% 保存结果
SSS.to_hdf("SSS.h5", "SSS")
# SSS = pd.read_hdf("SSS.h5", "SSS")

#%% -----------
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
os.chdir('D:\FinTech_2022') 
import _pickle as pickle
import time
import tensorflow as tf
import datetime

# 导入自定义函数包
from Functions import *

#%% 处理X_label
'''完整50个因子
'Alpha2', 'Alpha13', 'Alpha14', 'Alpha15', 'Alpha18', 'Alpha20',
       'Alpha34', 'Alpha46', 'Alpha65', 'Alpha66', 'Alpha88', 'Alpha106',
       'Alpha167', 'Alpha3', 'Alpha189', 'Alpha19', 'Alpha38', 'Alpha31',
       'Alpha52', 'Alpha93', 'Alpha110', 'Alpha118', 'Alpha126', 'Alpha127',
       'Alpha129', 'Alpha153', 'Alpha171', 'Alpha183', 'Alpha187', 'Alpha22',
       'Alpha53', 'Alpha58', 'Alpha59', 'Alpha69', 'Alpha71', 'Alpha86',
       'Alpha161', 'Alpha165', 'Alpha6', 'Alpha8', 'Alpha10', 'Alpha12',
       'Alpha17', 'Alpha37', 'Alpha41', 'Alpha175', 'Alpha120', 'Alpha185',
       'Alpha78', 'Alpha166'
'''
FT_ = SSS[['Stkcd','Date','Alpha2', 'Alpha13', 'Alpha14', 'Alpha15', 'Alpha18', 'Alpha20',
       'Alpha34', 'Alpha46', 'Alpha65', 'Alpha66', 'Alpha88', 'Alpha106',
       'Alpha167', 'Alpha3', 'Alpha189', 'Alpha19', 'Alpha38', 'Alpha31',
       'Alpha52', 'Alpha93', 'Alpha110', 'Alpha118', 'Alpha126', 'Alpha127',
       'Alpha129', 'Alpha153', 'Alpha171', 'Alpha183', 'Alpha187', 'Alpha22',
       'Alpha53', 'Alpha58', 'Alpha59', 'Alpha69', 'Alpha71', 'Alpha86',
       'Alpha161', 'Alpha165', 'Alpha6', 'Alpha8', 'Alpha10', 'Alpha12',
       'Alpha17', 'Alpha37', 'Alpha41', 'Alpha175', 'Alpha120', 'Alpha185',
       'Alpha78', 'Alpha166']]
del SSS
# 缺失值、异常值、标准化
FT_ = treat_X_label(FT_)
# 添加日期序号
FT_ = get_date_id(FT_, 'Date') 

#%% S 去中性化
# 将因子特征关于行业虚拟变量、市值、市值二次项跑回归取残差值。
# 导入行业数据集
IND = pd.read_csv('IND.csv') # 'Stkcd', 'Ind'
# 导入市值数据集（以总资产自然对数计算）
Size = pd.read_csv('Size.csv')

# 合并行业
FT_ = pd.merge(FT_,IND,on='Stkcd',how='left')

# 生成年月
FT_['ymd'] = FT_['Date'].astype(str).replace('\-', '', regex=True)
Size['ymd'] = Size['Accper'].astype(str).replace('\-', '', regex=True)
FT_['ym'] = FT_['ymd'].astype(int) // 100
Size['ym'] = Size['ymd'].astype(int) // 100
Size = Size[['Stkcd','ym','Size', 'Size2']]

# 先精确匹配
FT_ = pd.merge(FT_,Size,on=['Stkcd','ym'],how='left')

# 释放
del IND,Size

# 再前向填充
for i in ['Size','Size2']:
    FT_[i] = FT_.groupby(['Stkcd'])[i].bfill()
    print(str(nn),str(datetime.datetime.now()))
    nn = nn - 1

#%% S 生成行业虚拟变量
namespace = globals()

# 生成A-S
Inds = []
for i in range(65,84):
    Inds.append(chr(i))

# 生成虚拟变量
for i in Inds:
    FT_['Dum_%s' % (i)] = np.where(FT_['Ind'].str.contains(i, regex=False)==True,1,0)
    print(i)

# 收集虚拟变量名称，回归备用
Inds_add = ""
for i in Inds:
    Inds_add = Inds_add + "+" + 'Dum_%s' % (i)
    
#%% S 回归取残差
import statsmodels.formula.api as sm
# FT_.columns
Tobe = ['Alpha2', 'Alpha13', 'Alpha14', 'Alpha15', 'Alpha18', 'Alpha20',
       'Alpha34', 'Alpha46', 'Alpha65', 'Alpha66'] # 需要被处理的因子，手动粘贴过来

#%% S
import datetime
from itertools import product
yearmons = FT_['ym'].unique()
nn = len(Tobe)*len(yearmons)
R1 = pd.DataFrame(columns=['Stkcd','Date','Fac','Factor_neu'])
for Fac,Ym in product(Tobe, yearmons):
    length = get_len(FT_, Ym)
    R2 = pd.DataFrame(columns=['Stkcd','Date','Fac', 'Factor_neu'],
                    index=np.arange(length))
    R2['Fac'] = Fac
    try:
        R2['Stkcd'], R2['Date'], R2['Factor_neu'] = My_Resid(FT_, Fac,Ym)
    except ValueError:
        pass
    R1 = pd.concat([R1,R2]).dropna()
    nn = nn - 1
    if nn%10 == 0:
        print(str(nn) +',' +str(datetime.datetime.now()))


#%% 加入回报率数据
SSS = pd.read_hdf("SSS.h5", "SSS")
FT_ = pd.concat([FT_,SSS['RET']],axis=1)
del SSS 
    
#%% 划分训练集与测试集

# 切换保存路径
os.chdir('D:\FinTech_2022\halfway') 


date_num = len(FT_['Date'].unique())
ID_Days = list(range(1,date_num+1))
interval = 100
split_rate = 0.3
nn=0
start_days,end_days,split_days = [],[],[]
for i in tqdm(range(interval,date_num,30),ncols=100):
    nn=nn+1
    
    start_day = ID_Days[i-interval]
    end_day = ID_Days[i]
    split_day = ID_Days[int(i-interval*split_rate)]
    
    start_days.append(start_day)
    end_days.append(end_day)
    split_days.append(split_day)
    
    ret_fac = FT_[(FT_.ID_Day<=end_day)&
                  (FT_.ID_Day>=start_day)]
    
    # 处理Y_label
    train_set,test_set = treat_Y_label(ret_fac, 
                                       split_day)
    
    # 特征工程
    X_train, train_label, X_test, test_label,stock_list = FeatureSelect(ret_fac,train_set,test_set)
    
    # 依次保存
    save_file(X_train, 'X_train', nn)
    save_file(train_label, 'train_label', nn)
    save_file(X_test, 'X_test', nn)
    save_file(test_label, 'test_label', nn)
    save_file(stock_list, 'stock_list', nn)
    
    # print(len(ret_fac))
    # mix_fac = mix_fac.append(get_pred(ret_fac, end_day))
print('训练集与测试集划分 Finish!')

#%% 划分训练集与验证集
for i in tqdm(range(1,nn+1),ncols=100):
    
    # 导入
    X_train = load_file('X_train', i)
    train_label = load_file('train_label', i)
    X_test = load_file('X_test', i)
    test_label = load_file('test_label', i)
    stock_list = load_file('stock_list', i)
    
    # 划分
    np.random.seed()
    indices = np.random.permutation(X_train.shape[0])
    split = int(X_train.shape[0]*0.9)
    training_idx,test_idx = indices[:split],indices[split:]
    tran_tran,tran_vali = X_train[training_idx,:],X_train[test_idx,:]
    tran_label,vali_label = train_label[training_idx,:],train_label[test_idx,:]
    # print('训练集与验证集划分 Finish!')   
    
    # 保存
    save_file(training_idx, 'training_idx', i)
    save_file(test_idx, 'test_idx', i)
    save_file(tran_tran, 'tran_tran', i)
    save_file(tran_vali, 'tran_vali', i)
    save_file(tran_label, 'tran_label', i)    
    save_file(vali_label, 'vali_label', i)  

#%% 滚动训练

mix_fac = pd.DataFrame()

# 滚动训练模型
for i in tqdm(range(1,nn+1),ncols=100):
    X_train = load_file('X_train', i)
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
                
                # 加入GAN
                gan = GAN([n]+i+[2],tran_tran, tran_label, tran_vali, vali_label,j,k)
                acc = gan.train()
                pred,real = gan.get_pred(tran_vali, vali_label)
                evals.append(real_a(pred,real))                
                pred, real = gan.get_pred(X_test, test_label)
                evals2.append(real_a(pred,real)) # 按顺序保存
                
                preds.append(pred)      
                end = time.time()
                print(portfolio[-1],acc,end-start) # 验证集准确率
    
    # 确定最优参数，取出相应模型的预测值，评判标准是多头预测胜率
    win_df = pd.DataFrame(evals, index = portfolios, columns = np.linspace(0.05,0.3,6))
    win_df['avg'] = np.mean(win_df.values, axis=1)
    win_df.reset_index(inplace=True)
    final_pred = preds[win_df['avg'].idxmax()]
    
    end_day = end_days[i-1]
    
    # 给出股票名+日期
    cross_section = pd.DataFrame(final_pred[:,0],
                                 columns = [end_day],
                                 index = stock_list).T # 当期预测上涨的概率
    cross_name = 'cross_section' + '_' + str(i) + '.csv'
    cross_section.to_csv(cross_name,encoding='utf_8_sig') 
    
    mix_fac = mix_fac.append(cross_section)

#%%
# 最后拼在一起
FT2 = pd.concat([FT_.iloc[:,:2],FT_.iloc[:,-1]],axis=1)

#%%
# 透视列
FT2 = FT2.drop_duplicates(subset = ['Stkcd','Date'], keep = 'last')
FT2["ymd"] = FT2["Date"].astype(str).replace('\-', '', regex=True).astype(int)
FT2 = FT2[FT2['ymd']>=20180101]

#%%
df3 = FT2.set_index(['Stkcd','Date'])
factor_file = df3.INDEX.unstack(0)

factor_file.to_csv('factor_file_hyk.csv',
                   encoding='utf_8_sig') 

data_t = (factor_file-factor_file.mean())/(factor_file.max()-factor_file.min())
data_t = data_t.fillna(0)
data_t.to_csv('data_t_hyk.csv',
                   encoding='utf_8_sig') 

#%% 导入沪深300收益率数据
bench_1 = pd.read_excel('IDX_Idxtrd_1.xlsx')
bench_2 = pd.read_excel('IDX_Idxtrd_2.xlsx')
bench = pd.concat([bench_1.iloc[2:,1:],bench_2.iloc[2:,1:]])


#%%
bench = bench.rename(columns = {
    'Idxtrd01':'Date',
    'Idxtrd08':'RET'})
bench = date_change2(bench, 'Date')

#%% 保存结果
bench.to_hdf("bench.h5", "bench")
# bench = pd.read_hdf("bench.h5", "bench")

#%%
bench['Stkcd'] = 'benchmark'


#%%

df = pd.concat([FT_[['Stkcd','Date','RET']],bench])
df["ymd"] = df["Date"].astype(str).replace('\-', '', regex=True).astype(int)
df = df[df['ymd']>=20180101]


#%% 得到回报率文件
df = df[['Stkcd','Date','RET']].drop_duplicates(subset = ['Stkcd','Date'], keep = 'last')
# df = date_change(df, 'Date') 
ret_file = df.set_index(['Stkcd','Date']).RET.unstack(0)
ret_file  = ret_file.iloc[:-1,:]
ret_file.to_csv('ret_file_hyk.csv',
                   encoding='utf_8_sig') 


#%% 回测结果
# 输出收益率,因子特征,方向
poet_sim = Poet_Sim(ret_file,factor_file,direct=-1) 

# 0.1表示十分位,0.2表示五分位
rets = poet_sim.get_strategyret(decile=0.1) 

# 每次只能输入一个收益率Series
results = poet_sim.get_performance(rets['l_ret'],
                                   rf_annual=0,
                                   show=False) 

# 第一个是单边印花税,第二个是双边券商佣金
rets_c = poet_sim.get_cost_tur(fee1=0.001,fee2=0.0005) 

#%% 分年度返回策略评价指标
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










