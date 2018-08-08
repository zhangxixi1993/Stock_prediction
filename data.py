from lib.gftTools import gftIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_index():
    # 对不同公司的股票数进行排序,并提取公司位置
    num=[]
    y_dic = gftIO.zload('y_5.pkl')  # 获取数据,字典格式
    y_value = y_dic['stock_close_price']  # 获取键股票价格的值,每一列代表一家公司的股票价格
    y_value = np.array(y_value)  # 转换为二维数组
    for i in range(np.shape(y_value)[1]):   # 遍历每一家公司
        y = list(y_value[:,i] )  # 第i家的股票价格
        y = list(np.nan_to_num(y)) # nan变为0
        for j in y:
            if j == 0:
                y.remove(j)  # 去除nan
        temp=len(y)
        num.append(temp)   # 获得处理后每家公司的股票数,存在列表num
    index1=np.argsort(num)   # 对所有公司的股票进行升序排序,获得公司的位置存入index1
    return index1, y_value

def Data():

    index, y_value = find_index()
    temp = int(input('please input a number(0-3436):'))  # 数字越大取到的公司的股票数越多
    y_value = y_value[:, index[temp]]  # 获取指定的公司股票
    data1 = np.array(y_value)  # 将其转换为数组格式
    data1 = np.nan_to_num(data1)  # 将空值替换为0
    data1 = list(data1)  # 转换为列表
    for i in data1:  # 去除列表中的0元素
        if i == 0:
            data1.remove(i)
    return data1


def Return(close_price):
    data1=[]
    for i in range(len(close_price)-1):
        j=i+1
        temp=(close_price[j]-close_price[i])/close_price[i]  # 得到return,即涨跌百分比
        data1.append(temp)   # 将涨跌信息保存到data列表
    return data1

'''


def find_index():
    # 对不同公司的股票数进行排序,并提取公司位置
    num=[]
    y_dic = gftIO.zload('y_5.pkl')  # 数据列表
    y_value = list(y_dic.values())[1]  # 获取字典的第2个属性值,也就是所有公司的index
    for k in range(728):
        data = y_value.iloc[:, k]  # 选取一家公司的股票价格
        data = np.array(data)  # 将其转换为数组格式
        temp = 0
        for i in data:
            if i > 0:
                temp += 1
        num.append(temp)
    index=num.index(max(num))
    return index,num
a=find_index()
print(a)
print(Data())

'''

