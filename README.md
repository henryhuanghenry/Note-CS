# Note-CS
This repository contains some notes and materials that I took while learning some CS courses. 
Created by Henry Huang
---
## 1. 文件夹MOOC-PKU-tensorflow
### 1.1 文件夹内容
文件夹为MOOC 北大的人工智能实践：tensorflow的慕课学习资料。
**主要内容为：**
- 官方源代码和PPT和笔记
- 本人学习的笔记
- 本人学习过程中的代码
- mnist.npz 数据集
- to be continued..
### 1.2 mnist数据集本地加载方法
mnist.npz 把仓库中的数据集下载到本地后,的载入方法如下：
```python
import tensorflow as tf
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data(r'本地路径\mnist.npz')#加载本地数据集
#记得一定要加mnist.npz，否则程序会认为是load数据集到指定的路径，
#依然会下载下来，更有甚者会报错permission denied 就是他需要新建一个文件但是没有权限。
```
