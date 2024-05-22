import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
import tools

batch_size, num_steps = 32, 35
train_iter, vocab = tools.load_data_time_machine(batch_size, num_steps)

"""
    总结：
        现代循环神经网络的核心思想主要是对隐藏状态的H进行优化
        优化维度：
            1. 更新的方式 如设置更新门、隐藏门，或如LSTM
            2. 设置更深层次的网络学习隐藏状态，以达到更优的隐藏状态
            3. 使用Attention技术提高学习的范围，或使用双向循环神经网络
            
"""
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        """
            用于简化初始化代码
        """
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    # 设置为元组同样是为了LSTM预留
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        # \* 符号代表对应位置元素点乘
        """
            如公式所示：
                1. 这里的Z(更新门)，R(重置门)在经历相应的运算后均要通过sigmoid激活函数，因此降低了模型在运算过程中因为累加梯度造成梯度爆炸的概率
                2. 这里的Z，R的引入，增加了可学习参数的数量
                3. 如果Z为零矩阵，则该模型退化为RNN；若Z矩阵为全1矩阵，则该模型不对隐藏层状态进行更新
        """
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = tools.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
tools.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
plt.show()