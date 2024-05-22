import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
from torch import nn
import tools
import matplotlib


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )


net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 随机将输入张量中的某些元素置零
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 这里使用池化层可以减少参数数量，并且减少过拟合 但是为什么？
    # 对每个通道都进行池化，因为最终的分类类别为十种，所以将输出通道数量设置为10
    # 因为nn.AdaptiveAvgPool2d((1, 1))会对每层通道进行池化，且最终结果不会合并，所以最终产生的答案是一个[10,1,1]的矩阵
    nn.Flatten())
    # 展平层用于获取答案
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
tools.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()

