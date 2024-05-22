import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l
import tools

batch_size, num_steps = 32, 35
train_iter, vocab = tools.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 5
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
# 第三个参数为层数，下一层的隐藏状态矩阵H既依赖于本层的H_t-1，同时又依赖于上一层的隐藏状态矩阵H_t。
# 深度学习网络是用于学习隐藏层状态矩阵H，在中间不产生任何的output输出，只在神经网络的最后一层执行同RNN一样的执行逻辑
model = tools.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
num_epochs, lr = 500, 2
tools.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)

plt.show()