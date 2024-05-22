import torch
from torch import nn
from d2l import torch as d2l

def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K))

X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
# 将卷积核的权重设置为K矩阵，这里的卷积核的权重就是进行运算的卷积核本身

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
#
tconv.weight.data = K
print(tconv(X))