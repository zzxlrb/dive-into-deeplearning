import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

import tools


def dropout_layer(X, dropout):
    """
    Dropout layer
    :param X: 张量
    :param dropout: 丢弃输入张量X的概率
    :return:
    """
    assert 0 <= dropout <= 1
    if dropout == 1.0:
        return torch.zeros_like(X)
    elif dropout == 0.0:
        return X
    flag = (torch.rand(X.shape) > dropout).float()
    # 这里的放缩是为了保持期望输出值不变，从而避免了训练过程中的偏差？（不是很理解）
    return flag * X / (1.0 - dropout)


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    # 这里表示Net类继承继承自nn.Module类，nn.Module是 PyTorch 中所有神经网络模型的基类
    def __init__(self, num_inputs, num_outputs,
                 num_hiddens1, num_hiddens2, is_training=True):
        """
            __init__()是初始化方法，在创建类的实例时自动调用，用于初始化对象的属性
            在初始化方法内部，使用 self 关键字来访问对象的属性
        """
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        """
            用于自定义强项传播计算方法
            且继承自nn.Model的类必须实现 forward方法
        """
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout，所以在定义模型的时候不需要传入暂退的概率
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
tools.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()

#简洁实现
#这里的Sequential的含义是按照定义的顺序依次执行每个层的前向传播操作
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))
#注意：在该中定义方式下，在进行测试时，需要将模型的模式设置为‘eval’模式，以确保模型中的任何随机操作行为都处于禁用状态

"""
    总结：
        nn.Sequential 与 类定义 用于定义神经网络是等价的
        但是与Sequential方法相比，使用类定义的方式可以更加灵活的定义神经网络的结构和前向传播的过程
        使得神经网络模型能够支持各种自定义的处理和转换，大大增强了模型的灵活性
"""

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);