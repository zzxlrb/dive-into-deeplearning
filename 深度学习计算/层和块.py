import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F


class Mymodel(nn.Module):
    # 自定义神经网络的方式如下
    def __init__(self):
        super(Mymodel, self).__init__()
        self.hidden1 = nn.Linear(20, 256)
        self.hidden2 = nn.Linear(256, 10)
        self.out = nn.Linear(10, 1)

    def forward(self, X):
        return self.out(F.relu(self.hidden2(F.relu(self.hidden1(X)))))


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            # OrderedDict类似于字典，但是会记住元素被添加的顺序，这样可以确保模块可以按照他们被添加到该类中的顺序进行迭代
            # 在_modules字典中，键必须为字符串类型
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            #这里的self._modules.values()返回的是字典中索引对应的值的迭代器
            X = block(X)
        return X

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        # 自定义类好处：允许进行新的架构组合，而具有更强的灵活性，即可控制在前向传播中的数据流
        # 可定义常数参数，实在则深度学习网络优化过程中常数参数不可被更新
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
