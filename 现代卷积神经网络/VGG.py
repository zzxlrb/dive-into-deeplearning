import torch
from torch import nn
from d2l import torch as d2l
import tools
import matplotlib.pyplot as plt


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        #这里的激活函数会应用该tensor的所有元素上，有助于网络学习到更复杂和抽象的特征
        #可以增加神经网络的表达能力，提高模型的泛化能力
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


# 列表中的元素包括卷积层的数量和输出通道数

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        # 这里让输入通道数等于输出通道数以确保模型的正确性

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
        #此处的全连接层将若干特征值转化为分类的个数
        #因为CrossEntropy函数在内部对模型的原始输出应用softmax，所以在模型的最后一层不需要添加softmax函数


ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# 这里将通道数减少，用于处理计资源有限的情况
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
tools.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

plt.show()
