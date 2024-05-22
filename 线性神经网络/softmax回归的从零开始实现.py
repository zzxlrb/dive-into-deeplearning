import numpy as np
import torch
from d2l import torch as d2l
from IPython import display

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 这里是之前已经实现了的数据加载函数
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, (num_inputs, num_outputs), requires_grad=True)
b = torch.normal(0, 0.01, (num_outputs,), requires_grad=True)


def softmax(X):
    # 这里其实是对函数定义不同导致的歧义
    expX = torch.exp(X)
    part = expX.sum(dim=1, keepdim=True)
    """
        dim=0则按列求和，反之则按行求和
        若keepdim=True，则返回值为行向量或列向量，反之则为退化后的一维数组
    """
    return expX / part
    """
        这里利用了广播机制：
            如果两个张量的维度数量不同，那么较小维度的张量会被自动扩展至与较大维度相同的维度数量。
            如果两个张量在某个维度上的大小不同，但其中一个张量的大小为 1，那么该维度上的大小为 1 的张量会被扩展至与另一个张量相同的大小。
            如果两个张量在某个维度上的大小都不为 1，且在该维度上的大小不同，那么广播操作会失败，抛出错误。
    """


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
    # 这里reshape一下使得28*28的图像转化为行向量


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])


# 前者表示的是代取元素的对应的行标，而后者对应的是代取元素对应的列标。其索引从 0-len-1，分别取出对应元素

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


y_hat = torch.tensor([[0.1, 0.3, 0.6],  # 第一个样本的预测概率分布
                      [0.3, 0.2, 0.5]])  # 第二个样本的预测概率分布
# 真实标签
y = torch.tensor([0, 2])

print(cross_entropy(y_hat, y))
# 尝试着打印一下就懂了
"""
    y_hat是每一个物品的被预测成为每一个类别的的一个二维矩阵
    而y中存放的值为对应类别的下标
    根据代价函数可知正确性
"""


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        # 在这里 axis=1 代表按行求最大值，返回的矩阵为列向量，每行数值对应着该行最大值的下标索引
    cmp = y_hat.type(y.dtype) == y
    # “y_hat.type(y.dtype)” 该语句将y_hat的数据类型转化为同y矩阵一样  用转化后的矩阵与y矩阵进行逐个元素比较 ，生成一个布尔型张量
    return float(cmp.type(y.dtype).sum())
    # 将bool类型的张量转化为整数类型的张量，其中True对应1，False对应0 求和后转为浮点数类型（便于计算）返回


class Accumulator:
    # 累加类
    # 在定义类方法时，第一个参数通常命名为self，可以更改为其他参数名
    # 在调用类方法时，Python会自动将调用该方法的实例作为第一个参数传递给类方法

    def __init__(self, n):  # 构造函数
        self.data = [0.0] * n

    def add(self, *args):
        # 注意这里加了*号，表示取出容器中的元素作为变量而不是把这个容器当作变量
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        # '__'双下划线在python中表示特殊方法或特殊属性，
        # __getitem__是一个特殊方法，用于实现对象的索引访问功能。可以简单的理解成实现了[]取数
        return self.data[idx]


def evaluate_accuracy(net, data_iter):  # @save
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
        """
            进入推理阶段（评估模式）时：
                为了使评估模式的而模型与训练产生的模型一致且稳定
                将模型设置为评估模式从而禁止一些具有随机性质的操作对模型进行更改
        """
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
        """
            在训练模式下，模型会启用特定的训练行为，例如：Dropout层
            同时模型将利用输入输入数据进行前向传播和反向传播，并更新模型的权重以进行训练
        """
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            """
                isinstance(object, classinfo)
                其中，object 是要检查的对象，classinfo 可以是类对象或者类对象组成的元组。
                如果 object 是 classinfo 类或其子类的一个实例，那么 isinstance() 返回 True，否则返回 False。
            """
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            """
                调用l.mean().backward()时，Pytorch会计算损失函数的平均值，并对平均值进行反向传播，
                并最终会得到每个参数再整个批次上的平均梯度
                故再训练过程中梯度计算不会受到批次大小的影响，使得模型更加收敛，训练过程更加稳定
            """
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            """
                调用l.sum().backward()时，Pytorch会计算损失函数的总和，并对总和进行反向传播，
                并最终会得到每个参数再整个批次上的总和梯度
            """
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:  # @save
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """
        xlabel: 横轴的标签，默认为 None。
        ylabel: 纵轴的标签，默认为 None。
        legend: 图例，默认为 None。图例用于标识每条线对应的含义。
        xlim: 横轴的范围，默认为 None。
        ylim: 纵轴的范围，默认为 None。
        xscale: 横轴的刻度，默认为 'linear'。可选值包括 'linear' （先行可都，数值等间隔线性增长）和 'log（对数刻度，数值按照对数尺度增长）'。
        yscale: 纵轴的刻度，默认为 'linear'。可选值包括 'linear' 和 'log'。
        fmts: 线条格式，默认为 ('-', 'm--', 'g-.', 'r:')。用于指定每条线的样式，例如实线、虚线、点划线等。
        nrows: 子图的行数，默认为 1。
        ncols: 子图的列数，默认为 1。
        figsize: 图的大小，默认为 (3.5, 2.5)。用于指定绘图的尺寸。
        """
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        """
            lambda匿名函数：
                语法结构： lambda 参数 : 表达式
            构造这么一个固定了的隐式函数，在add函数中会用到
        """
        self.X, self.Y, self.fmts = None, None, fmts
        # Python 中的类变量通常是通过在类的方法中使用 self. 来定义的，不需要显示定义类变量，并在类的所有实例之间共享

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            """
                __len__ 是一个特殊方法（魔术方法），用于返回对象的长度或大小
                hasattr(object, name) 用于检查一个对象是否具有指定的属性或方法
            """
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        # 清除当前子图上的所有绘图内容
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
            # 在图形对象的第一个子图上绘制多条线条
        self.config_axes()
        # lambda函数，调用则会按照初始化中规定的格式对图像进行调整
        display.display(self.fig)
        # 将图形对象显示在当前的输出环境中
        display.clear_output(wait=True)
        # 除当前输出单元格的内容，并等待下一个输出的到来，以保持输出的干净整洁。


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc
    # 断言：这里的作用时确保训练正确率与损失在规定范围内，如果有任何一个assert语句的条件不满足，则会引发异常从而使程序停止运行


def updater(batch_size):
    return d2l.sgd([W, b], 0.1, batch_size)


num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        # 仅获取一次数据样本和对应标签，并不需要遍历整个测试数据集
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


predict_ch3(net, test_iter)
