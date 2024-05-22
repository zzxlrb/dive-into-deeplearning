import torch
import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch import nn

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def train(lambd):
    w, b = init_params()
    # 初始化参数w，b
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    #匿名函数，可以理解为为函数起别名，使得代码更加简洁
    num_epochs, lr = 100, 0.003
    # 定义训练步数以及学习率
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
            """
                d2l框架实现的梯度下降函数
                需要注意的是这里的梯度需要除以batch_size
                如果不除以 batch_size，梯度下降更新的步长会受到批量大小的影响，从而导致对参数的更新过程不稳定
                造成收敛速度不稳定，优化效果不佳
            """
    print('w的L2范数是：', torch.norm(w).item())


train(lambd=3)
plt.show()

#简洁实现

def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    """
        {}是一个字典形式的参数设置，用于创建字典。
        在本案例中，{}中包含了两个键值对，每个键值对分别对应着参数设置的属性
        'weight_decay' 是优化器的一个属性，用于控制权重衰减的程度
    """
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)

    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())