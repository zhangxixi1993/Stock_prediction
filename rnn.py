'''
股票价格预测,数据长度为1857
选取前1500个数据进行训练,后357个数据进行预测.选取利用前10个数据预测后一个数据
'''
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from matplotlib.font_manager import FontProperties
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
import matplotlib.pyplot as plt
import data
import pandas as pd
# import EMD

learn = tf.contrib.learn   # tensorflow的神经网络库
HIDDEN_SIZE = 30  # 隐含层神经元数
NUM_LAYERS = 2  # 两层
TIMESTEPS = 10  #
TRAINING_STEPS = 10000  # 训练次数
BATCH_SIZE = 10 # 批次大小



def generate_data(seq):    # 生成时间序列
    X = []
    Y = []
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])
        Y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    return lstm_cell

def lstm_model(X, y):
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])
    predictions = tf.contrib.layers.fully_connected(output, 1, None)
    labels = tf.reshape(y, [-1])   # 实际值
    predictions = tf.reshape(predictions, [-1])   # 预测值
    loss = tf.losses.mean_squared_error(predictions, labels)   # 损失函数
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),optimizer="Adagrad", learning_rate=0.1)  # 训练器
    return predictions, loss, train_op              # 返回预测结果,损失和训练器
def RNN():
    train_X, train_y = generate_data(seq)  # 根据以上时间序列构造训练数据, train_X相当于1489个样本,每个样本长度为10,train_y为训练输出,为长度为1489的列向量
    test_X, test_y = generate_data(np.array(data2[TRAINING_EXAMPLES:len(data2)]))

    regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)


    predicted = []
    for pred in regressor.predict(np.array(test_X)):
        predicted.append([pred])  # 得到预测输出
  #  print(predicted)
    rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))  # 实际值与预测值的均方根误差
    print("Mean Square Error is:%f" % rmse)
    predicted_total=predicted
    return test_y,predicted_total

def EMD_RNN():
    x = np.array(data2)  # 将列表转为数组
    imf = EMD.emd(x)  # 将return进行EMD分解
    for i in range(np.shape(imf)[0]):
        data3 = imf[i]  # 取IMF分量
        seq = np.array(data3[0:TRAINING_EXAMPLES])  # 将时间序列存放到数组,前1500个数据训练
        train_X, train_y = generate_data(seq)  # 根据以上时间序列构造训练数据, train_X相当于1489个样本,每个样本长度为10,train_y为训练输出,为长度为1489的列向量

        test_X, test_y = generate_data(np.array(data3[TRAINING_EXAMPLES:len(data3)]))
        regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

        predicted = []
        for pred in regressor.predict(test_X):
            predicted.append([pred])  # 得到预测输出
        rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))  # 实际值与预测值的均方根误差

        print("Mean Square Error is:%f" % rmse)
        for j in range(len(predicted)):
            predicted_total[j] = predicted[j][0] + predicted_total[j]  # 汇总所有IMF的预测结果

    test_X, test_y = generate_data(np.array(data2[TRAINING_EXAMPLES:len(data3)]))
    return test_y,predicted_total

data1 = data.Data()    # 调用data函数得到股票价格
# data2 = data1
data2 = data.Return(data1)   # 调用data的return方法得到时间序列数据,返回return列表

TRAINING_EXAMPLES = int(len(data2)*0.8)  # 训练样本数

TESTING_EXAMPLES = len(data2)-TRAINING_EXAMPLES  # 测试样本
SAMPLE_GAP = 0.01
predicted_total = []
for i in range(TESTING_EXAMPLES - TIMESTEPS - 1):
    predicted_total.append(0)
seq = np.array(data2[0:TRAINING_EXAMPLES])  # 将时间序列存放到数组,前1500个数据训练
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir="Models/model_2"))
###########################


#####################


test_y, predicted_total = RNN()

#计算return的预测准确率
a=[]
b=[]
x=0
for i in test_y:
    if i[0]>=0:
        a.append(1)
    else:
        a.append(-1)
for i in predicted_total:
    if i>=0:
        b.append(1)
    else:
        b.append(-1)
for i in range(len(a)):
    if a[i]==b[i]:
        x+=1
print('涨跌预测准确率',x/len(a))

print('所有样本预测结果',predicted_total)
plot_predicted, = plt.plot(predicted_total,'b', label='pre_return')
plot_test, = plt.plot(test_y, 'r',label='real_return')
plt.legend([plot_predicted, plot_test], ['pre_return','real_return'])
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('sample')
plt.ylabel('return')
plt.title('真实值与预测值的对比')
plt.show()






