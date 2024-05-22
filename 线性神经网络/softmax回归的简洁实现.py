import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l
import tools

batch_size = 18
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


# 这里的第一层（也就是Flatten）的作用是将28*28的二维图层展开成为一维的向量
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=1, std=0.01)
        # mean为平均值，std为标准差
        nn.init.normal_(m.bias, std=0.01)


# 这里通过初始化函数，将神经网络中所有的线性层初始化为对应的值。
# 这里的特判是因为展平层中不包含任何参数，避免对其进行不必要的操作
net.apply(init_weights)
# 这里是将初始化函数应用到该神经网络的所有模块上

loss = nn.CrossEntropyLoss(reduction='none')
# 参数 reduction='none' 表示不对每个样本的损失值进行求和或平均，而是保留每个样本的损失值。

trainer = torch.optim.SGD(net.parameters(), lr=0.01)
num_epochs = 10

tools.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.plot()