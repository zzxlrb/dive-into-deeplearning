import random
import torch


def generate(w, b, cnt):
    X = torch.normal(0, 1, (cnt, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


def data_iter(batch_size, features, y):  #数据随机选择器，感觉写的还蛮巧妙地？
    cnt = len(features)
    indices = list(range(cnt))
    random.shuffle(indices)
    for i in range(0, cnt - 1, batch_size):
        batch_incides = indices[i:min(i + batch_size, cnt - 1)]
        yield features[batch_incides], y[batch_incides]


def line(w, features, b):
    return torch.matmul(features, w) + b


def loss_fn(y, y_pred):
    return torch.sum((y - y_pred) ** 2 / 2)


def sgd(params, lr, batch_size):
    with torch.no_grad(): #这里一定记得加 （虽然我也不知道具体是什么意思）
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


ture_w = torch.tensor([2, -3.4])
ture_b = 4.2
feature, y = generate(ture_w, ture_b, 1000)

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

batch_size = 10

lr = 0.03
epochs = 1000
for epoch in range(epochs):
    for X, lable in data_iter(batch_size, feature, y):
        l = loss_fn(line(w, X, b), lable)
        l.backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        if (epoch % 100 == 0):
            print(epoch, l)
print(ture_w - w, ture_b - b)

