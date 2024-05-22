import torch
import matplotlib.pyplot as plt
import numpy as np
from d2l import torch as d2l
from torch import nn
from torchvision import datasets, transforms
from torch.utils import data

batch_size, epochs = 256, 100
loss_array = []
accuracy_array = []

num_input, num_hidden, num_outputs, num_classes = 784, 256, 10, 10

train_dataset = datasets.MNIST(root='./Mnist', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./Mnist', train=False, download=True, transform=transforms.ToTensor())

mnist_train = data.DataLoader(train_dataset, batch_size, shuffle=True)
mnist_test = data.DataLoader(test_dataset, 16, shuffle=True)

net = nn.Sequential(nn.Flatten(), nn.Linear(num_input, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_outputs),
                    nn.Softmax(dim=1))


def init_weight(m):
    if (type(m) == nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.zeros_(m.bias)


net.apply(init_weight)

criterion = nn.CrossEntropyLoss()
trainer = torch.optim.Adam(net.parameters())


def accuracy(lable, target):
    result = (lable == target)
    return float(result.int().sum()) / float(len(lable))


for epoch in range(epochs):
    for X, y in mnist_train:
        net.zero_grad()
        outputs = net(X)
        lable = outputs.argmax(dim=1, keepdim=True)
        loss = criterion(outputs, y)
        loss.mean().backward()
        trainer.step()
    loss_array.append(loss.item())
    accuracy_array.append(accuracy(lable.reshape(1, -1).view(-1), y))

step = [i for i in range(epochs)]

plt.plot(step, loss_array, marker='v')
plt.plot(step, accuracy_array, marker='o')
plt.show()
fig, ax = plt.subplots(2, 8)

for X, y in mnist_test:
    outputs = net(X)
    lable = outputs.argmax(dim=1, keepdim=True)
    ax = ax.flatten()
    i = 0
    for a in ax:
        a.imshow(X[i].view(28, 28))
        a.set_title(str(lable[i].item()))
        i += 1
    break

plt.show()
net.eval()

all_predictions = []
all_results = []
for x, y in test_dataset:
    all_results.append(y)
    outputs = net(x)
    label = outputs.argmax(dim=1, keepdim=True)
    all_predictions.append(label.item())

print('Accuracy on test: {:.2f}%'.format(accuracy(torch.tensor(all_predictions), torch.tensor(all_results)) * 100))