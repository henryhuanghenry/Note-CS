# TF学习-TF基本操作

Created by Yusheng Huang

[TOC]



---

课程链接：https://www.bilibili.com/video/BV1Di4y1N7fb?p=2

github地址：https://github.com/henryhuanghenry/Machine-Learning-Collection

## 1. 创建张量

### 1.1 创建普通的张量

```python
x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
print(x)

x = tf.constant([[1, 2, 3], [4, 5, 6]], shape=(2, 3))
print(x)
```

### 1.2 张量的矩阵化创建

```python
x = tf.eye(3) #单位矩阵
print(x)

x = tf.ones((4, 3)) #1矩阵
print(x)

x = tf.zeros((3, 2, 5))#0矩阵
print(x)

x = tf.random.uniform((2, 2), minval=0, maxval=1)#随机创建一个矩阵，均匀分布随机
print(x)

x = tf.random.normal((3, 3), mean=0, stddev=1)#随机创建矩阵，正态分布随机
print(tf.cast(x, dtype=tf.float64))#使用cast可以强制类型转换
# tf.float (16,32,64), tf.int (8, 16, 32, 64), tf.bool

x = tf.range(9)#创建一个序列的张量
x = tf.range(start=0, limit=10, delta=2)
print(x)
```

## 2. 张量的数学操作

假设有x和y两个张量

```python
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])
```

### 2.1加

```python
z = tf.add(x, y)
z = x + y#上下等价
```

### 2.2减

```python
z = tf.subtract(x, y)
z = x - y#上下等价
```

### 2.3 element wise乘

```python
z = tf.multiply(x, y)
z = x * y#上下等价
```

### 2.4 点乘

```python
z = tf.tensordot(x, y, axes=1)
```

### 2.5矩阵乘法

```python
x = tf.random.normal((2, 3))
y = tf.random.normal((3, 2))
z = tf.matmul(x, y)
z = x @ y#上下等价
```

### 2.6除

```python
z = tf.divide(x, y)
z = x / y#上下等价
```

### 2.7 幂

```python
z = x ** 5
```



## 3.张量的索引访问

### 3.1基本

切片访问的操作与python的基本操作相同:

- 右不包含
- 从0开始
- [:]访问全部
- [：：2]2可以指定步长，把2改成-1可以反向遍历

### 3.2指定index集合的访问

```python
indices = tf.constant([0, 3])#指定index集
x_indices = tf.gather(x, indices)#抽取指定的index的元素
```



## 4.张量的reshape

```python
x = tf.range(9)

x = tf.reshape(x, (3, 3))#reshape

x = tf.transpose(x, perm=[1, 0])#交换矩阵的维度
```

