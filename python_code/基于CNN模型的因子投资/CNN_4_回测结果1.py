#%% ===========================
#%% Part 4: 回测结果1
#%% ===========================

# 在繁重的模型训练之后，可以在定制的回测引擎中比较模型。
# 针对中国股市设计了回溯引擎，考虑了流动性和可交易性。
# 回溯测试引擎排除了IPO股票和触及价格限制的股票，因为这些股票几乎没有流动性。
# 该引擎尽可能接近真实的交易环境，以提供公平和可靠的回测结果。

#%% 
import numpy as np
import pandas as pd
# from pandas import to_datetime as tdt, Timestamp as tp
from tqdm import tqdm
from pathlib import Path
# import matplotlib.pyplot as plt

import os
os.chdir('D:\FinTech_2022') 

class BackTest:
    """
    A class designed for machine learning model backtesting

    Parameters
    ----------------
    dfScore : DataFrame, containing the predicted stock return.

    begt: datetime, backtest begin date. Can be specified or empty.

    endt: datetime, backtest begin date. Can be specified or empty.

    n: int, the number of stocks selected in daily portfolio.

    fee: int, transaction fee. Default 0.2%
    ----------------
    """
    # initialize class
    def __init__(self, dfScore, begt=np.nan, endt=np.nan, n=30, freq=10, fee=0.002):
        self.dfScore = dfScore
        self.begt = begt
        self.endt = endt
        self.n = n
        self.freq = freq
        self.fee = fee
        self._set_begin_end()
        self._init_price_matrix()
        self._init_index()

    # Set begin date and end date
    def _set_begin_end(self):
        if np.isnan(self.begt):
            self.begt = self.dfScore.index.get_level_values(0).min()
        if np.isnan(self.endt):
            self.endt = self.dfScore.index.get_level_values(0).max()

    # Load market data
    def _init_price_matrix(self):
        print('###############Intializing Market Information###############')
        data = pd.read_parquet("data.parquet").set_index(['date', 'stockid'])
        data = data.loc[
            pd.IndexSlice[self.begt:self.endt, :], ['vwap', 'adjfactor', 'isnew', 'isst', 'istrade', 'limit']]
        data['vwap_adj'] = data.vwap * data.adjfactor
        self.price_matrix = data['vwap_adj'].unstack()
        self.status_matrix = data[['isnew', 'isst', 'istrade', 'limit']]

    # Load index data
    def _init_index(self):
        indexdata = pd.read_parquet("marketindex.parquet").set_index('date')
        self.indexdata = indexdata.loc[self.begt:self.endt]

    # get all market trading days
    def gettradedate(self):
        tradedate = self.indexdata.index
        return tradedate[(tradedate >= self.begt) & (tradedate <= self.endt)]

    # get index for a specific day
    def getindex(self, t):
        try:
            indexclose = self.indexdata.loc[t, 'close']
            base = self.indexdata.iloc[0, 0]
        except:
            raise TypeError('Stock Index Reading Error！')

        return (indexclose / base)

    # get target portfolio for a specific day
    def gettradestock(self, t):
        tscore = self.dfScore.loc[pd.IndexSlice[t, :]]
        # delete unavailable stocks
        tprice = self.status_matrix.loc[pd.IndexSlice[t, :]]
        # tprice = tprice[(tprice.istrade == 1) & (tprice.isnew == 0) & (tprice.isst == 0) & (tprice.limit != 1)]
        tprice['score'] = tscore.score
        tprice.dropna(subset=['score'])
        tprice = tprice.sort_values(by='score', ascending=False)
        return tprice.head(self.n).index

    # calculate maximum drawdown of a portfolio
    def maxdrawdown(self, rarr):
        marr = [rarr[0:i + 1].max() for i in np.arange(len(rarr))]
        ddarr = rarr / marr - 1
        return (ddarr.min())

    # calculate Sharpe ratio of a portfolio
    def sharpe_ratio(self, rarr, mark):
        return ((rarr.loc[:, mark].mean() - 0.0001) / rarr.loc[:, mark].std())

    # backtest function
    def backtest(self):
        # get all market trading days
        tradedates = self.gettradedate()
        # get all trade days for our portfolio
        orderdates = pd.Series(self.dfScore.index.get_level_values(0).unique())
        orderdates = orderdates[np.arange(len(orderdates)) % self.freq == 0]
        # get portfolio traadelist
        tradelist = {t: self.gettradestock(t) for t in orderdates}
        # initialize netvalue DataFrame()
        netvalue = pd.DataFrame(index=tradedates,
                                columns=['netvalue', 'benchmark', 'return',
                                         'benchmark return', 'excess return'])
        netvalue.index.name = 'date'

        print('\n###############Backtest Start###############')
        for i, idate in tqdm(enumerate(tradedates)):
            if i == 0:
                netvalue.loc[idate, 'netvalue'] = 1 * (1 - self.fee / 2)
            else:
                # get current portfolio
                lasttradedate = orderdates[orderdates < idate].max()
                holdings = tradelist[lasttradedate]
                # calculate daily return
                if idate in orderdates:
                    netvalue.loc[idate, 'netvalue'] = netvalue.loc[lasttradedate, 'netvalue'] * \
                                                      (self.price_matrix.loc[idate, holdings] / \
                                                       self.price_matrix.loc[lasttradedate, holdings]).mean() * (
                                                              1 - self.fee)
                else:
                    netvalue.loc[idate, 'netvalue'] = netvalue.loc[lasttradedate, 'netvalue'] * \
                                                      (self.price_matrix.loc[idate, holdings] / \
                                                       self.price_matrix.loc[lasttradedate, holdings]).mean()
            # get benchmark performance
            netvalue.loc[idate, 'benchmark'] = self.getindex(idate)
            # calculate return and excess return
            netvalue['return'] = netvalue.netvalue / netvalue.netvalue.shift(1).fillna(method='bfill') - 1
            netvalue['benchmark return'] = netvalue['benchmark'] / netvalue['benchmark'] \
                .shift(1).fillna(method='bfill') - 1
            netvalue['excess return'] = netvalue['return'] - netvalue['benchmark return']

        self.netvalue = netvalue
        return (self.netvalue)

    # portfolio evaluation
    def evaluation(self):
        if not hasattr(self, 'netvalue'):
            self.backtest()

        res = pd.DataFrame(columns=['yearmonth', 'return', 'annual return',
                                    'maximum drawdown', 'Sharpe', 'winrate',
                                    'excess return', 'annual excess return'])

        alpha = self.netvalue.copy(deep=True)
        alpha.loc[:, 'year'] = alpha.index.year
        alpha.loc[:, 'yearmonth'] = alpha.index.year * 100 + alpha.index.month
        alpha['fullreturn'] = alpha.netvalue / alpha.netvalue.iloc[0] - 1
        alpha['fullexcessreturn'] = alpha.netvalue / alpha.netvalue.iloc[0] - \
                                    alpha.benchmark / alpha.benchmark.iloc[0]

        yearmonth = alpha.loc[:, 'yearmonth'].unique()
        for ni in np.arange(len(yearmonth)):
            data = alpha[alpha.yearmonth == yearmonth[ni]].sort_values(by='date', ascending=True)
            wincount = len(data[data['excess return'] > 0])

            if len(data) <= 1:
                break
            res.loc[ni, 'yearmonth'] = str(yearmonth[ni])
            res.loc[ni, 'return'] = (1 + data['return']).product() - 1
            res.loc[ni, 'annual return'] = res.loc[ni, 'return'] * 12
            res.loc[ni, 'maximum drawdown'] = self.maxdrawdown(data.fullreturn + 1)
            res.loc[ni, 'Sharpe'] = self.sharpe_ratio(data, 'return') * np.sqrt(252)
            res.loc[ni, 'winrate'] = wincount / len(data)
            res.loc[ni, 'excess return'] = (1 + data.iloc[-1, :]['fullexcessreturn']) / (
                        1 + data.iloc[0, :]['fullexcessreturn']) - 1
            res.loc[ni, 'annual excess return'] = res.loc[ni, 'excess return'] * 12

        res.loc[ni + 1, 'yearmonth'] = 'Total'
        res.loc[ni + 1, 'return'] = (1 + alpha.iloc[-1, :]['fullreturn']) / (1 + alpha.iloc[0, :]['fullreturn']) - 1
        res.loc[ni + 1, 'annual return'] = res.loc[ni + 1, 'return'] * 12 / len(yearmonth)
        res.loc[ni + 1, 'maximum drawdown'] = self.maxdrawdown(alpha.fullreturn + 1)
        res.loc[ni + 1, 'Sharpe'] = self.sharpe_ratio(alpha, 'return') * np.sqrt(252)
        res.loc[ni + 1, 'winrate'] = len(alpha[alpha['excess return'] > 0]) / len(alpha)
        res.loc[ni + 1, 'excess return'] = (1 + alpha.iloc[-1, :]['fullexcessreturn']) / (
                    1 + alpha.iloc[0, :]['fullexcessreturn']) - 1
        res.loc[ni + 1, 'annual excess return'] = res.loc[ni + 1, 'excess return'] * 12 / len(yearmonth)
        self.valuation = res
        return (self.valuation)

    # plot portfolio and benchmark performance
    def plot(self):
        if not hasattr(self, 'netvalue'):
            self.backtest()
        netvalue = self.netvalue
        netvalue[['netvalue', 'benchmark']].plot()


if __name__ == "__main__":
    config = {
        'file': "Baseline_2019-01-01_2020-12-31.parquet"
    }

    prediction = pd.read_parquet(Path("predictions")/config['file'])
    bt = BackTest(prediction)
    bt.backtest()

    print(bt.netvalue)
    print(bt.evaluation())


#%% CNN
os.chdir('D:/FinTech_2022')
cnn_predict = pd.read_parquet("D:/FinTech_2022/MLF_2021_Final_Project-main/predictions/CNN_%s_%s.parquet" % (begt, endt))
bt_cnn = BackTest(cnn_predict, n=50, freq=10)
print ("CNN performance:")
display(bt_cnn.evaluation())
bt_cnn.plot()

