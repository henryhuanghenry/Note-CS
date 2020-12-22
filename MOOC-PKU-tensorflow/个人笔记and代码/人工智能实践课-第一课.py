
# coding: utf-8

# # PKU-人工智能实践课-第一课-莺尾花代码
# 
# Created by Henry Huang
# ----
# 代码实现。

# # 1.使用莺尾花数据集，进行神经网络初识。

# 导入包

# In[325]:


import tensorflow as tf
import numpy as np


# 定义超参数

# In[326]:


epoch=1000
lr=0.05
train_loss_result=[]
test_acc=[]


# 导入数据集
# 
# （遇到错误ImportError: cannot import name 'datasets'）--似乎是没有安装sklearn导致的

# In[327]:


import sklearn.datasets as datasets


# In[328]:


x_data = datasets.load_iris().data #返回iris数据集所有输入特征
y_data = datasets.load_iris().target #返回iris数据集所有标签


# 对数据进行乱序

# In[329]:


np.random.seed(115) # 使用相同的seed，使输入特征/标签一一对应
np.random.shuffle(x_data)
np.random.seed(115)
np.random.shuffle(y_data) 
tf.random.set_seed(115)


# 分开训练集和测试集

# 知识点就是<font color=red>Python的正负索引</font>
# 
# 列表中每个数都有一个正索引和负的索引。
# 
# 切片时候，step为正决定从左往右取数，step为负决定从右往左取数。
# 
# **正索引：**左边第一个数索引是0，然后从左开始递增。
# 
# **负索引:**右边第一个数索引是0，然后从右边开始递减。

# In[330]:


x_train = x_data[:-30]#这个操作结果跟[0:150-30]的效果是一样的
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]


# 然后分batch

# In[331]:


train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
#train_db=tf.cast(train_db,dtype=tf.double)
#test_db=tf.cast(test_db,dtype=tf.double)


# 定义一个4*3的全连接网络
# 
# <font color=red>这里必须将所有的都转化为float32才能进行矩阵乘法运算。float64不行，难道是怕乘法溢出？</font>

# In[332]:


w1 = tf.Variable(tf.random.truncated_normal([4,3], stddev=0.1, seed=2))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=2))
w1=tf.cast(w1,dtype=tf.float32)
b1=tf.cast(b1,dtype=tf.float32)


# In[333]:


for epoch in range(epoch): #数据集级别迭代
    for step, (x_train, y_train) in enumerate(train_db): #batch级别迭代
        loss_all=0
        with tf.GradientTape() as tape:# 记录梯度信息
            x_train=tf.cast(x_train,dtype=tf.float32)
            y_out=tf.matmul(x_train,w1) + b1 #计算神经网络的输出
            y_poss=tf.nn.softmax(y_out)#化为概率分布
            y_true=tf.one_hot(y_train,depth=3)#对真实值独热编码
            loss=tf.reduce_mean(tf.square(y_poss-y_true))#计算loss
            loss_all+=loss.numpy()
            
        grads = tape.gradient(loss, [w1,b1])
        w1.assign_sub(lr * grads[0]) #参数自更新
        
        b1.assign_sub(lr * grads[1])
    train_loss_result.append(loss_all/4)
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    
    total_correct=0
    total_number=0
    for x_test, y_test in test_db: 
        x_test=tf.cast(x_test,dtype=tf.float32)
        y = tf.matmul(x_test, w1) + b1 # y为预测结果
        y = tf.nn.softmax(y) # y符合概率分布
        pred = tf.argmax(y, axis=1) # 返回y中最大值的索引，即预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype) #调整数据类型与标签一致
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum (correct) # 将每个batch的correct数加起来
        total_correct += int (correct) # 将所有batch中的correct数加起来
        total_number += x_test.shape [0]
        acc = total_correct / total_number
        test_acc.append(acc)
    print("test_acc:", acc)


# 可视化

# In[334]:


from matplotlib import pyplot as plt
plt.title('Acc Curve') # 图片标题
plt.xlabel('Epoch') # x轴名称
plt.ylabel('Acc') # y轴名称
plt.plot(range(epoch+1),test_acc, label="$Accuracy$") # 逐点画出test_acc值并连线
plt.legend()
plt.show()
plt.title('Loss Curve') # 图片标题
plt.xlabel('Epoch') # x轴名称
plt.ylabel('Loss') # y轴名称
plt.plot(range(epoch+1),train_loss_result, label="$Loss$") # 逐点画出test_acc值并连线
plt.legend()
plt.show()

