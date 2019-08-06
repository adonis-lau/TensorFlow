#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 训练模型根据输入的摄氏度输出华氏度

import logging

import numpy as np
import tensorflow as tf

# 设置日志输出级别
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# 摄氏度（输入  Feature，特征）
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
# 华氏度（输出  Labels，标签）
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
# 一对Feature和Labels组成Example，训练样本

# 循环输出展示一下
for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

# 定义层级。层级：神经网络中相互连接的节点集合
# Dense，密集层
# input_shape=[1]，模型的输入是1，也就是单个密集层
# units指定了这个层级将有多少内部变量
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# 创建一个包含层级（l0）的模型。模型：神经网络的表示法
model = tf.keras.Sequential([l0])

# 也可以直接把层级放入模型中创建，例如：model = tf.keras.Sequential([ tf.keras.layers.Dense(units=1, input_shape=[1]) ])

# 编译模型，指定两个参数：损失函数 和 优化器
# TF在训练过程中使用损失函数和优化器寻找最佳模型
# mean_squared_error 均方误差
# Adam(0.1) 中的 0.1 称为学习速率
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# 训练模型。模型调用fit方法，传入 摄氏度和华氏度输入-输出样本、周期（epochs）设置为500
# epochs，周期，是指对我们看到的样本进行一次完整的迭代。我们有7个样本，所以模型一共将训练3500个样本
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

# 用图表显示训练周期
# X轴，Epoch Number，训练周期
# Y轴，Loss Magnitude，损失值
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

# 传入摄氏度值，模型会返回相应的的华氏度值
print(model.predict([100.0]))

# 输出密集层的内部变量，即权重
# 结果应该会有两个内部变量，他们应该与 摄氏度转华氏度的公式（f=c×1.8+32）中的因子几乎相等
print("These are the layer variables: {}".format(l0.get_weights()))


# 看看三层模型
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0]))
print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))
print("These are the l0 variables: {}".format(l0.get_weights()))
print("These are the l1 variables: {}".format(l1.get_weights()))
print("These are the l2 variables: {}".format(l2.get_weights()))