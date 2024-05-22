import torch
from torch import nn
from torch.nn import functional as F

#加载和保存张量
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
X = torch.rand(size=(10,2))

#加载和保存模型
net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU())
torch.save(net, 'net-file')
tran_net = torch.load('net-file')

#加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
Y_clone = clone(X)
Y_clone == Y

"""
    除去上述的实例，torch.save()函数亦可保存：
        包括但不限于字典、列表、元组、numpy数组等类型变量
"""

