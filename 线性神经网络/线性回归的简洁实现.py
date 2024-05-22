import torch
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([3, 0.23])
true_b = 20
# 这个函数的功能应该是自动生成带噪声的样本点
features, lables = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    # 构造函数，将传入的数据列表或者数据源组转化为TensorDataset
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    # 进一步将dataset打乱拆分，创建用于批量加载数据的迭代器


batch_size = 20
data_iter = load_array((features, lables), batch_size, is_train=True)
#在 PyTorch 中，Sequential 类是一个容器类，用于将多个层（layers）按顺序组合在一起，形成一个神经网络模型。
# 它提供了一种简洁的方式来定义神经网络模型，特别适用于线性堆叠的网络结构。

net = nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
#这里的weight和bias其实就是在机器学习中的斜率和截距
#只不过用于这里的输入特征值的个数不再唯一，所以再使用斜率这个名词不恰当
#通过_结尾的方法将参数替换，从而初始化参数
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(),lr=0.03)
#net.parameters()是net中所有需要优化的参数的集合
epochs=1000
for epoch in range(epochs):
    for X,y in data_iter:
        y_pred = net(X)
        l = loss(y_pred, y)
        trainer.zero_grad()
        #默认情况下，反向传播计算得到的梯度会在每次调用backward()方法时累加而不是重新赋值
        #这代表着在每一次进行更新之前，都必须清空之前计算出来的梯度，避免重复计算造成的错误
        l.backward()
        trainer.step()
    #这里是计算在总体样本集上，代价函数的值
    if(epoch%100==0):
        l = loss(net(features), lables)
        print(f'epoch {epoch}, loss {l:f}')
w=net[0].weight.data
b=net[0].bias.data
print(true_b,true_w)
print(b,w)