import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图"""
    d2l.use_svg_display()
    # 该函数应用于D2L书记的代码示例中，指示系统使用SVG格式来显示图形
    # SVG(Scalable Vector Graphics)是一种矢量图形格式，能够以矢量形式存储图像，从而使图像在缩放时不会失真
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    # 共享轴刻度是指在绘制子图时，让该轴的所有刻度是相同的，不会因为数据范围的不同而产生偏差
    # 不要压缩空白的维度，是保留维度的完整性，而不会被压缩为低维数组。
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    # 画图没看懂！


attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
# 创建一个10x10 的单位矩阵，并将其重新变化形状为一个四维张量
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
plt.show()
