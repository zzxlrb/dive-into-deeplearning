import torch
from d2l import torch as d2l
import torch.nn as nn
import torch.nn.functional as F


def corr2d(X, K):
    a, b = K.shape
    x, y = X.shape
    Y = torch.zeros(size=(x - a + 1, y - b + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + a, j:j + b] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(size=(kernel_size, kernel_size)))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])

Y = corr2d(X, K)
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
X.requires_grad_(True)

lr = 0.03

for i in range(50):
    y_hat = conv2d(X)
    loss = ((Y - y_hat) ** 2).sum()
    conv2d.zero_grad()
    loss.backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    print(X.grad)
    print(conv2d.weight.grad)
    #loss.backward()会计算损失函数关于所有参与干损失函数计算的参数的梯度(梯度打开的参数)，并将这些梯度累积到各个参数的 .grad 属性中
    print(f'epoch{i+1} mean:{loss}')
print(conv2d.weight)
