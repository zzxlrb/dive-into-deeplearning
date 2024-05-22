import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
import numpy as np
from torch import nn

import tools

num_input, num_hidden_output, num_output = 784, 256, 10
net = nn.Sequential(nn.Flatten(), nn.Linear(num_input, num_hidden_output), nn.ReLU(),
                    nn.Linear(num_hidden_output, num_output))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.ones_(m.weight)
        nn.init.ones_(m.bias)


net.apply(init_weights)

batch_size, lr, epochs = 128, 0.1, 10
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

test_iter, train_iter = d2l.load_data_fashion_mnist(batch_size)

tools.train_ch3(net, train_iter, test_iter, loss, epochs, optimizer)

plt.show()