'''
预测sin函数的输出
假设一个时间序列长度为11000.用前10000个样本预测后1000个数,重构前10000个训练数据,每10个一组,接下来的一个为输出.
设时间序列为x(t),即取x(0-9)为第一组输入,x(10)为其对应输出;然后x(1-10)为第二组输入,x(11)为其对应输出,以此类推.....
测试时候同样重构预测序列,输入其输入得到预测值
'''
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import DATA1
import data
from matplotlib.font_manager import FontProperties
# import matplotlib as mpl
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
#mpl.use('Agg')
import matplotlib.pyplot as plt

learn = tf.contrib.learn   # tensorflow的神经网络库
HIDDEN_SIZE = 30  # 隐含层神经元数
NUM_LAYERS = 2  # 两层
TIMESTEPS = 10  #
TRAINING_STEPS = 10000  # 训练次数
BATCH_SIZE = 32  # 批次大小

TRAINING_EXAMPLES = 1500  # 训练样本数
TESTING_EXAMPLES = 357  # 测试样本
SAMPLE_GAP = 0.01

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
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir="Models/model_2"))
#data1 = DATA1.data()    # 调用data函数得到股票价格
#data2 = data1

#data2 = data.Return(data1)   # 调用data的return方法得到时间序列数据,返回return列表
data2=[]
for i in range(1300):
    data2.append(random.randint(1,10))

seq= np.array(data2[0:1200]) # 产生时间序列,长度为10000,范围0-100

train_X, train_y = generate_data(seq)   # 根据以上时间序列构造训练数据, train_X相当于9989个样本,每个样本长度为10,train_y为训练输出,为长度为9989的列向量
print(train_y)
print(train_X)
test_X, test_y = generate_data(np.array(data2[1200:1300]))

regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

predicted=[]
for pred in regressor.predict(test_X):
    predicted.append([pred])          # 得到预测输出

rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))   # 实际值与预测值的均方根误差
print("Mean Square Error is:%f" % rmse[0])

plot_predicted, = plt.plot(predicted, 'ro', label='predicted')
plot_test, = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted','real_sin'])
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('样本点')
plt.ylabel('函数值')
plt.title('真实值与预测值的对比')
plt.show()
