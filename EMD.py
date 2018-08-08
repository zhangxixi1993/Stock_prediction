import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
import pandas as pd
import data

# 判定当前的时间序列是否是单调序列
def ismonotonic(x):
    max_peaks=signal.argrelextrema(x,np.greater)[0]
    min_peaks=signal.argrelextrema(x,np.less)[0]
    all_num=len(max_peaks)+len(min_peaks)
    if all_num>=3:
        return False
    else:
        return True

# 寻找当前时间序列的极值点
def findpeaks(x):
    '''
     df_index=np.nonzero(np.diff((np.diff(x)>=0)+0)<0)
     u_data=np.nonzero((x[df_index[0]+1]>x[df_index[0]]))
     df_index[0][u_data[0]]+=1
     return df_index[0]
    '''
    return signal.argrelextrema(x,np.greater)[0]    # 返回序列的极大值点的index
# 判断当前的序列是否为 IMF 序列
def isImf(x):
    N=np.size(x)
    pass_zero=np.sum(x[0:N-2]*x[1:N-1]<0)  # 过零点的个数
    peaks_num=np.size(findpeaks(x))+np.size(findpeaks(-x))  # 极值点的个数
    if abs(pass_zero-peaks_num) > 1:
        return False
    else:
        return True
# 获取当前样条曲线
def getspline(x):
    N=np.size(x)
    peaks=findpeaks(x)
#     print ('当前极值点个数：',len(peaks))
    peaks=np.concatenate(([0],peaks))
    peaks=np.concatenate((peaks,[N-1]))
    if(len(peaks)<=3):
        '''
          if(len(peaks)<2):
#         peaks=np.concatenate(([0],peaks))
#         peaks=np.concatenate((peaks,[N-1]))
#         t=interpolate.splrep(peaks,y=x[peaks], w=None, xb=None, xe=None,k=len(peaks)-1)
#         return interpolate.splev(np.arange(N),t)
        '''
        t=interpolate.splrep(peaks,y=x[peaks], w=None, xb=None, xe=None,k=len(peaks)-1)
        return interpolate.splev(np.arange(N),t)
    t=interpolate.splrep(peaks,y=x[peaks])
    return interpolate.splev(np.arange(N),t)
#     f=interp1d(np.concatenate(([0,1],peaks,[N+1])),np.concatenate(([0,1],x[peaks],[0])),kind='cubic')
#     f=interp1d(peaks,x[peaks],kind='cubic')
#     return f(np.linspace(1,N,N))
# 经验模态分解方法
def emd(x):
    imf=[]
    while not ismonotonic(x):
        x1=x
        sd=np.inf
        while sd>0.1 or (not isImf(x1)):
            s1=getspline(x1)
            s2=-getspline(-1*x1)
            x2=x1-(s1+s2)/2
            sd=np.sum((x1-x2)**2)/np.sum(x1**2)        # sd的更新
            x1=x2
        imf.append(x1)   # 加入IMF
        x=x-x1
    imf.append(x)   # 添加最后的残余分量
    return imf



'''
data1 = data.Data()
data2 = data.Return(data1)  # 调用data函数得到时间序列数据
data = list(data2)
x = np.array(data)
imf1 = emd(x)
print('imf的大小为:',np.shape(imf1))

plt.plot(imf1[8])
plt.show()
'''


