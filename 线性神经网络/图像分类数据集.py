import torch
import numpy as np
import torchvision
from d2l import torch as d2l
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import timer

# transforms提供了一系列常用的图像处理操作
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='./FashionMNIST', train=True, download=False, transform=trans)
mnist_test = torchvision.datasets.FashionMNIST(root='./FashionMNIST', train=False, download=False, transform=trans)

print(mnist_test[0][0].shape)

# def get_fashion_mnist_labels(labels):
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]
#
#
# # 这个函数做的就是将labels中的数字转化为对应的文本
# datasets = data.DataLoader(mnist_train, batch_size=256, num_workers=4)
#
#
# # data.DataLoader支持并发操作
# def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
#     figsize = (num_cols * scale, num_rows * scale)
#     # 这里的行数和列数指的是子图的行数与列数，这里默认省去子图的大小，而这里的scale代表着对子图大小的缩放
#     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
#     # 这里的_代表着不关心的变量（可以理解为占位），而这里的axes里储存的是二维的子图
#     # 这里的figsize参数规定的是整张图的大小而非每个子图的大小
#     axes = axes.flatten()
#     # 将二维数组展开，便于访问
#     # 展开原则：索引顺序从左至右，从上至下排列
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         """
#             zip函数：
#                 zip函数用于合并两个或多个可迭代对象（例如元组、列表、集合）
#                 在合并时，zip函数将多个可迭代对象对应下标的元素合并为一个元组
#                 tips：如果传入的可迭代对象的长度不同，zip函数会截断较长的对象
#             enumerate函数：
#                 enumerate函数用于所有可迭代对象，支持在循环中同时返回元素对应的索引以及元素本身
#         """
#         if torch.is_tensor(img):
#             # 图片张量
#             ax.imshow(img.numpy())
#             # imshow()函数可以直接接受Numpy数组作为输入，但不能直接接受张量
#             """
#                 图像在pytorch中以张量的形式表示，张量的维度对应于图像的通道、高度和宽度。
#                 张量以多维数组的形式存贮在内存中，并且需要使用pytorch提供的相关函数进行处理
#             """
#         else:
#             # PIL图片
#             ax.imshow(img)
#             # imshow()函数可以直接接受PIL图片对象作为输入
#             """
#                 PIL图片是通过PIL库加载的图像对象，通常以文件的形式存在，可以使用PIL库的函数加载和处理
#             """
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     return axes
#
#
# X, y = next(iter(data.DataLoader(mnist_train, batch_size=16)))
# axes = show_images(X.reshape(-1, 28, 28), 2, 8, titles=get_fashion_mnist_labels(y))
# # 这里需要将X reshape是因为 生成的ax的个数为16，所以需要有16张图片与其对应，这里的重新分配就是为了避免zip函数产生截断
# plt.show()
# """
#     matplotlib绘图逻辑：
#         在使用matplotlib绘制图像时，调用图像绘制函数不会立即显示图像，而是将生成的图像对象添加至绘图区域中
#         而只有在调用plt.show()该函数时，matplotlib才会真正的将绘图区域中的图像对象显示出来
#         plt.show()函数的作用时将所有已经创建的绘图区域一起显示出来
#     关于绘制图像叠加：
#         绘图区域中会保存已绘制的图像，但是当绘制新的图像时便会累加至旧有图像。换言之，一个绘图区域只能存一张图（这里的图为一个大范围的定义）
#         如果想在一个绘图区域同时存有多个图像，则可以使用subplots函数
#     优点：好处多多？
# """
# timer = d2l.Timer()
# timer.start()
# train_iter = data.DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=4)
# timer.stop()
#
#
# def load_data_fashion_mnist(batch_size, resize=None):  # @save
#     # 前情提要：transforms提供了一系列常用的图像处理操作
#     trans = [transforms.ToTensor()]
#     # 这里的trans被赋值为一个操作列表
#     if resize:
#         trans.insert(0, transforms.Resize(resize))
#         # 如果resize被指定，则在操作列表的第一个位置插入重塑操作
#     trans = transforms.Compose(trans)
#     # 这句代码创建一个Compose对象，用于将多个转换操作串联起来，从而一次队图像数据进行处理（可以理解为一个定义好的自动化操作序列）
#     mnist_train = torchvision.datasets.FashionMNIST(
#         root="../data", train=True, transform=trans, download=True)
#     mnist_test = torchvision.datasets.FashionMNIST(
#         root="../data", train=False, transform=trans, download=True)
#     return (data.DataLoader(mnist_train, batch_size, shuffle=True,
#                             num_workers=4),
#             data.DataLoader(mnist_test, batch_size, shuffle=False,
#                             num_workers=4))
#                             # 注意这里捏：不知道什么错误，这里一指定num_workers就报错？ 我服了