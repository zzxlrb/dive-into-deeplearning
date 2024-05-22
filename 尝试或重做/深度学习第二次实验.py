import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

sigma = 1e-8
loss_array = []
weight_array = []
bias_array = []

x_train = torch.tensor([[8.], [3.], [9.], [7.], [16.], [05.], [3.], [10.], [4.], [6.]], dtype=torch.float32,
                       requires_grad=True)
y_train = torch.tensor([[0], [0], [1], [0], [1], [0], [0], [1], [0], [0]], dtype=torch.float32, requires_grad=True)
x_test = torch.tensor([[5.], [4.5], [9.8], [8.], [22.], [17.], [3.], [19.], [20], [30]], dtype=torch.float32,
                      requires_grad=True)
y_test = torch.tensor([[0], [0], [0], [1], [1], [1], [0], [1], [1], [1]], dtype=torch.float32, requires_grad=True)

batch_size = 8
learning_rate = 0.005
epochs = 1000

net = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())


def cross_entropy_loss(output, target):
    return sum(-(target * torch.log(output + sigma) + (1 - target) * torch.log(1 - output + sigma)))


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.ones_(m.weight)
        nn.init.ones_(m.bias)


net.apply(init_weight)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

for i in range(epochs):
    data_loader = d2l.load_array([x_train, y_train], batch_size, True)
    for x, y in data_loader:
        optimizer.zero_grad()
        y_pred = net(x)
        loss = cross_entropy_loss(y_pred, y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        bias_array.append(net[0].bias.data.numpy()[0])
        weight_array.append(net[0].weight.data[0].numpy()[0])
        loss_array.append(cross_entropy_loss(net(x_train), y_train).detach().numpy()[0])
step = [i for i in range(1, epochs + 1)]
plt.plot(step, loss_array)
plt.show()
answer = (net(x_test).detach().numpy() > 0.5).astype(int)
for i in range(len(answer)):
    print(y_test[i].detach().numpy()[0].astype(int), answer[i][0])
plt.show()
"""
现在可以做的事情有两个
    1.实现三维图象的绘制
    2.尝试调用不同的优化函数进行优化，比如RMSprop等等
"""
