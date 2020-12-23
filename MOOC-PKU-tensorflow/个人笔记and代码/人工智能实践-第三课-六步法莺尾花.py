
# coding: utf-8

# # 人工智能实践-第三课-六步法莺尾花
# 
# Created by Henry Huang
# 
# 主要旨在，部分不看老师视频的情况下，能不能用六步法复现一个莺尾花。
# 
# ---

# 第一步：导入模块

# In[28]:


import tensorflow as tf
from sklearn import datasets
import numpy as np


# 第二步：弄好训练集和测试集

# In[29]:


x_train=datasets.load_iris().data
y_train=datasets.load_iris().target
#因为测试集从训练集中划分，因此我们不需要自己再分割
np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)


# 第三步：写一个全连接模型(3个神经元)

# In[30]:


model= tf.keras.models.Sequential ([ 
    #就只有一个全连接层的神经网络，激活函数为relu，使用l2正则化
    tf.keras.layers.Dense(3, activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())
]) #描述各层网络


# 第四步：配置训练方式

# In[31]:


model.compile(optimizer =tf.keras.optimizers.SGD (lr=0.1),loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics = ['sparse_categorical_accuracy'] )


# 第五步：训练模型

# In[32]:


model.fit(
    x_train, y_train, 
    batch_size=32, epochs=500, 
    validation_split=0.2,
    validation_freq=20
)


# 第六步：打印网络结构和参数统计

# In[33]:


model.summary()


# 离谱，本来想写一个不用softmax的，把logistic那里改成true就行了。结果这个数据集还是用softmax正确率高。
