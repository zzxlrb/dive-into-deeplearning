import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
import tools

# 简洁实现

batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = tools.load_data_time_machine(batch_size, num_steps)
# 通过设置“bidirective=True”来定义双向LSTM模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = tools.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 1
tools.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

plt.show()