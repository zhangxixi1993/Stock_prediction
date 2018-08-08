import pandas as pd
from lib.gftTools import gftIO
import numpy as np
class StockCleaning:
    def __init__(self, frequency = 'daily', list_remove_days=0, fillna_method = 'median', fillna_value=None, fillna_lookback = 0):
        ## obtaining data
        # 2010.1.1-2017.8.30
        # y: 'stock_suspend_date', 'index_weight', 'benchmark', 'stock_list_date', 'stock_close_price',  90mb

        y_dic = gftIO.zload('Data/y_5.pkl')
        stock_close_price = y_dic['stock_close_price']

        stock_resampled = self.resample(stock_close_price, frequency) #(91, 3437)

        stock_list_removed = gftIO.zload('Data/list_date.pkl').asColumnTab().dropna() # 3442
        list_filtered = self.listed_filter(stock_resampled, stock_list_removed, list_remove_days)

        stock_suspend = gftIO.zload('Data/stock_suspend.pkl')
        stock_suspend_filtered = self.suspended_filter(list_filtered, stock_suspend)  # [91 rows x 2637 columns]

        stock_rtn = self.simple_return(stock_suspend_filtered) # [86 rows x 2566 columns]
        self.stock_cleaned = self.fillna(stock_rtn,fillna_method,fillna_value,fillna_lookback) # (86, 2566)

    def resample(self, data, frequency):

        if isinstance(data, gftIO.GftTable):
            df = data.asMatrix()
            if sum(df.count()) == df.shape[1]:
                df = data.asColumnTab().dropna()
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise Exception("Can not resample data of type:" + str(type(data)))

        df_freq = self.sort_freq(df, frequency)

        return df_freq


    def sort_freq(self, df, frequency):
        df.sort_index(inplace=True)

        # sort by frequency
        if frequency == 'monthly':
            freq_sorted = df.asfreq('BM', method='ffill')  # last working day of the month, fill na with last observation
        elif frequency == 'weekly':
            freq_sorted = df.resample('W-FRI').last()  # last working day of the week, fill na with last observation
        elif frequency == 'daily':
            bdays = pd.bdate_range(df.index[0], df.index[-1])
            freq_sorted = df.reindex(bdays, method='ffill')  # daily data, ffill
        return freq_sorted



    # 剔除上市不到一年的股票
    # listed_filter 1CAE433BFD3749E5A03DBD87E0C44D50

    # 去除上市不到remove_days的股票

    def listed_filter(self, stocks, listed_dates, remove_days):
        import datetime
        stock_df = stocks
        stock_list_date = listed_dates

        del_datepoint = stock_df.index[-1] - datetime.timedelta(days=remove_days)

        stock_del_df = stock_list_date[stock_list_date['idname'] > del_datepoint]
        stock_del_ls = stock_del_df['variable'].tolist()
        stock_filter_ls = list(filter(lambda x: x not in stock_del_ls, list(stock_df.columns)))
        stock_df_filtered = stock_df.filter(items=stock_filter_ls)

        return stock_df_filtered




    # 剔除样本最后一天及之后停牌的股票
    # suspended_filter (including cleaning): B38706EB4758492CB1193234705FD935
    # seperate suspend data cleaning choice: F999A73DA1E34D48B53C96A6FF7F5130

    import pandas as pd

    def suspended_filter(self, stocks, suspend_stock):
        suspend_stock_dic = self.suspend_data_cleaning(suspend_stock)

        stock_df = stocks

        stock_suspend_ls = list(set(suspend_stock_dic.keys()) & set(stock_df.columns))

        stock_del_ls = []

        for stock in stock_suspend_ls:
            if suspend_stock_dic[stock] >= stock_df.index[-1]:
                stock_del_ls.append(stock)

        stock_filter_ls = list(filter(lambda x: x not in stock_del_ls, list(stock_df.columns)))
        stock_df_filtered = stock_df.filter(items=stock_filter_ls)

        return stock_df_filtered

    def suspend_data_cleaning(self, data):
        df_dropna = data.asColumnTab().dropna()
        df_group_0 = df_dropna.groupby('value').get_group(0)
        groups = df_group_0.groupby('variable', squeeze=True)
        dic={}
        for name, group in groups:
            dic[name]=group.idname.iloc[-1] # name: stock, idname: dates
        return dic


    # simple return  33CBBB5CC97049EE8E28F4702510C1A5
    def simple_return(self, price):
        stock_price = price
        stock_rtn = stock_price.pct_change()

        # dropna of empty symbols
        stock_rtn.dropna(axis=1, how='all', inplace=True)
        #stock_rtn.dropna(how='all', inplace=True)

        return stock_rtn



    # fillna 8586D47551DA46398126484263169AF9

    # methods: 'median', 'mean', 'ffill', 'bfill'
    # filled by input value
    # lookback = -1 meaning None

    def fillna(self, data, method, fillna_value=None, lookback=None):
        stocks = data
        if method == 'median':
            stocks.fillna(value=stocks.median(), inplace=True)

        elif method == 'ffill' and 'bfill':
            if lookback < 0:
                lookback = None
            elif lookback == 0:
                raise Exception('Fillna method of %s limit cannot be 0.' % method)
            stocks.fillna(method=method, limit=lookback, inplace=True)

        elif method == 'mean':
            stocks.fillna(value=stocks.mean(), inplace=True)
        else:
            stocks.fillna(value=fillna_value, inplace=True)

        return stocks


data=StockCleaning()
data=data.stock_cleaned

