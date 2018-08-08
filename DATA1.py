from lib.gftTools import gftIO
import pandas as pd
import numpy as np
def data():
    y_dic = gftIO.zload('Data/y_5.pkl')  # 获取数据,字典格式
    y_value = list(y_dic.values())
    y_value=y_value[0].iloc[:,1500]
    y_value=list(y_value)
    y_value=list(np.nan_to_num(y_value))
    x=[]

    for i in y_value:
        if i > 0:
            x.append(i)
    return x

def Return(close_price):
    close_price=np.array(close_price)
    close_price2=np.roll(close_price,1)
    return (close_price[1:]-close_price2[1:])/close_price2[1:]