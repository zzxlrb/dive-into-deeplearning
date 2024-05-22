import torch
from torch import nn


def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 好聪明的写法
    return Y.reshape(Y.shape[2:])


conv2d = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(2, 2),stride=(3,2))
# 注意这里的padding是两边填充的行列数，总填充的行列数应该为2*padding
X = torch.randn((8, 8))

Y = comp_conv2d(conv2d, X)
print(Y.shape)
