import torch
from d2l import torch as d2l
from torch import nn
import torch.nn.functional as F

#不带参数
class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()
        #super().__init__()
        #在单继承中，以上两种构造方没有显著差别，但是为了代码的清晰和可读性，建议使用第一种方法
    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

#带参数
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        """
            Parameter是一个类，继承自Tensor，是专门用于表示模型参数的类，用于创建一个可学习的参数
            而在自定义模型中，只有模型参数被初始化为Parameter类的实例，才能被parameter()方法捕获
        """
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

net = MyLinear(10,10)
print(net.parameters())