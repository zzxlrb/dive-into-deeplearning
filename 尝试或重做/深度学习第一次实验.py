import torch
from torch.utils import data
import numpy as np
from torch import nn
from d2l import torch as d2l

features = torch.tensor([[8], [3], [9], [7], [16], [5], [3], [10], [4], [6]], dtype=torch.float, requires_grad=True)
labels = torch.tensor([[30], [21], [35], [27], [42], [24], [10], [38], [22], [25]], dtype=torch.float,
                      requires_grad=True)
#注意在自定义数据的时候，一定要严格规定Tensor中数据的数据类型，并且开启记录梯度


batch_size = 10


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    #这里*的作用是是将一个包含多个张量的列表或元组展开，将其中的每个张量作为单独的参数传递给构造函数
    #而如果不适用*，则传入的是整个元组，参数数量不匹配
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)



data_iter = load_array([features, labels], batch_size)
#首先要记得，这里要传的参数是特征值和实际值的合集
#其次，这里传入元组或是列表均可


net = nn.Sequential(nn.Linear(1, 1))
net[0].weight.data.fill_(1)
net[0].bias.data.fill_(1)

lr = 0.005

#感觉从这往下跟傻子一样，也没思考要干啥，全给我提示出来了？？？？

loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr)


epochs = 10000
for epoch in range(epochs):
    for x, y in data_iter:
        y_pred = net(x)
        l = loss(y, y_pred)
        #这里注意不要与函数名相同，否则会报错
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print('Epoch:', epoch, '\tLoss:', loss(net(features),labels))
w = net[0].weight.data
b = net[0].bias.data
print(w, b)
