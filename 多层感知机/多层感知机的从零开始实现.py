import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l
import tools


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
"""
nn.Parameter类为torch.Tensor的子类，主要用于表示模型的可学习参数。
特点：当其实例化对象被赋值给nn.Model的属性时，它们会自动注册为模型的参数，并在迭代过程中被优化器所更新，并且可在模型的parameters() 或 named_parameters() 方法中访问
"""
params = [W1, b1, W2, b2]


# 创建变量列表


def relu(X):  # 定义激活函数RELU
    template = torch.zeros_like(X)
    return torch.max(X, template)
    # 这里记得是返回一个Tensor而不是单纯的调用max函数返回一个最大值矩阵


def net(X):
    X = X.reshape(-1, num_inputs)  # 这里看看是不是会有bug，等会儿回过头来确认一下
    H = relu(torch.matmul(X, W1) + b1)  # 这里“@”代表矩阵乘法
    return (torch.matmul(H, W2) + b2)


loss = nn.CrossEntropyLoss(reduction='none')
"""
    代价函数为CrossEntropyLoss不再多余赘述
    这里的参数reduction表示是否对最终的样本损失值降维，按什么方式降维（降维可以理解成从多维数据中求出其整个的特征值）
    当 reduction 设置为 'none' 时，表示不进行降维操作，即返回每个样本的损失值。
    当 reduction 设置为 'mean' 时，表示对每个样本的损失值进行求均值操作，最终返回一个标量值，表示整个 batch 的平均损失。
    当 reduction 设置为 'sum' 时，表示对每个样本的损失值进行求和操作，最终返回一个标量值，表示整个 batch 的总损失。
"""
num_epochs, lr = 20, 0.01
updater = torch.optim.SGD(params, lr=lr)
tools.train_ch3(net=net, train_iter=train_iter, test_iter=test_iter, loss=loss, updater=updater, num_epochs=num_epochs)
plt.show()
