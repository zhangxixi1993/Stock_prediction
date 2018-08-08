import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import DATA1
import MEMD
import stock_cleaning
from math import sqrt

def generate_data(seq):
    '''
    :param seq: 输入为时间序列列表
    :return: 输出时间序列的重构结果,包含作为输入的多维数组和作为输出的一维数组
    '''
    X = []
    y = []
    for i in range(len(seq) - timesteps -1):
        X.append(seq[i: i+timesteps])
        y.append(seq[i+timesteps])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
# 获取一个batch_size大小的数据
def get_batches(X, y, batch_size=32):
    '''
    :param X: 输入多维数组
    :param y: 输入对应的一维数组
    :param batch_size: 批次大小
    :return: 返回一个批次的数据
    '''
    for i in range(0, len(X), batch_size):
        begin_i = i
        if i+batch_size<len(X):
            end_i=i+batch_size
        else:
            end_i=len(X)
        yield X[begin_i:end_i], y[begin_i:end_i]

epochs = 20      # 迭代次数
timesteps = 20   # 每个训练样本的长度
lstm_size = 100   # 隐藏层神经元
lstm_layers = 2  # 隐藏层个数
batch_size = 20  # 批次大小

x = tf.placeholder(tf.float32, [None, timesteps, 1], name='input_x')
y_ = tf.placeholder(tf.float32, [None, 1], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

data=stock_cleaning.StockCleaning()
data=data.stock_cleaned
company=2000
data=np.array(data)[:,company]

training_examples = int(len(data) * 0.8)           # 选取数据的80%进行训练,其余进行测试
testing_examples = len(data) - training_examples   # 测试数据个数
_, Test_y = generate_data(data[training_examples:len(data) + 1])  # 保留初始的测试值
imf=MEMD.emd(data)    # 进行经验模态分解

Results=[]                                         # 初始化一个全为0的列表,用于保存最终结果
for i in range(testing_examples - timesteps - 1):
    Results.append(0)
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)     # 有lstm_size个单元
drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)      # 添加dropout

def lstm_cell():
    '''
    :return: # 定义添加隐含层函数
    '''
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])

outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)    # 进行forward，得到隐层的输出
outputs = outputs[:, -1]     # 在本问题中只关注最后一个时刻的输出结果，该结果为下一个时刻的预测值

predictions = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=tf.tanh)   # 定义输出层, 输出值[-1,1]，因此激活函数用tanh
cost = tf.losses.mean_squared_error(y_, predictions)       # 定义损失函数
optimizer = tf.train.AdamOptimizer().minimize(cost)        # 定义优化步骤

session = tf.Session()
for i in range(np.shape(imf)[0]):
    data=list(imf[i])   # 获取分解结果的每一个分量,生成数据
    train_x, train_y = generate_data(np.array(data[0:training_examples]))
    test_x, test_y = generate_data(np.array(data[training_examples:len(data) + 1]))

    with session.as_default() as sess:
        # 初始化变量
        tf.global_variables_initializer().run()
        iteration = 1
        for e in range(epochs):
            for xs, ys in get_batches(train_x, train_y, batch_size):
                # xs[:,:,None] 增加一个维度，例如[20, 10] ==> [20, 10, 1]，为了对应输入
                # 同理 ys[:,None]
                loss, _ = sess.run([cost, optimizer], feed_dict={x: xs[:, :, None], y_: ys[:, None], keep_prob: .5})
                if iteration % 20 == 0:
                    print('Epochs:{}/{}'.format(e, epochs),'Iteration:{}'.format(iteration),'Train loss: {:.8f}'.format(loss))
                iteration += 1
    with session.as_default() as sess:
        results = sess.run(predictions, feed_dict={x: test_x[:, :, None], keep_prob: .5})  # 测试结果
    for j in range(len(results)):
        Results[j] = Results[j]+results[j]   # 融合所有分量的预测结果
results=Results

plt.plot(results,'r', label='predicted')
plt.plot(Test_y, 'g--', label='real_return')
plt.title('company %d'%company)
plt.legend()
plt.show()
a=[]
b=[]
xx = 0
# 计算涨跌预测准确率

for i in Test_y:
    if i >= 0:
        a.append(1)
    else:
        a.append(-1)
for i in results:
    if i >= 0:
        b.append(1)
    else:
        b.append(-1)
for i in range(len(a)):
    if a[i] == b[i]:
        xx += 1
print('EMD+RNN涨跌预测准确率', xx / len(a))
MAE=sum(abs(Test_y-np.array(results).flatten()))/len(Test_y)
RMSE=sqrt(sum((Test_y-np.array(results).flatten())**2)/len(results))
print('MAE:',MAE,'RMSE',RMSE)