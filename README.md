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
- mnist 数据集
- fashion_mnist 数据集
- to be continued..
### 1.2 mnist数据集本地加载方法
mnist.npz 把仓库中MOOC文件夹里面的mnist.npz数据集下载到本地后,的载入方法如下：
```python
import tensorflow as tf
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data(r'本地路径\mnist.npz')#加载本地数据集
#记得一定要加mnist.npz，否则程序会认为是load数据集到指定的路径，
#依然会下载下来，更有甚者会报错permission denied 就是他需要新建一个文件但是没有权限。
```

### 1.3 fashion_mnist 数据集数据集本地加载方法
mnist.npz 把仓库中的数据集下载到本地后,的载入方法如下：
1. 从仓库中MOOC的文件夹里面找到fashion的文件夹下载文件到本地,有能力而不信任我提供的文件的同学可以去官方下载地址：
  - https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
  - https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
  - https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
  - https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
2. 下载之后,随便放到一个文件夹里面，复制文件夹路径。然后拷贝下面我改的函数:
```python
import os
import numpy as np
import gzip
def load_data_fromlocalpath(input_path):
  """Loads the Fashion-MNIST dataset.
  Modified by Henry Huang in 2020/12/24.
  We assume that the input_path should in a correct path address format.
  We also assume that potential users put all the four files in the path.

  Load local data from path ‘input_path’.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
  ]

  paths = []
  for fname in files:
    paths.append(os.path.join(input_path, fname))  # The location of the dataset.


  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)
```
3. 使用下面的函数就能调用数据集了：
``` python
(x_train,y_train),(x_test,y_test)=fashion.load_data(r'本地地址')#加载本地数据集
```
---

## 2.文件夹Note-TFonlineTutorials

该文件夹保存了一个online的tensorflow的tutorials的学习笔记
