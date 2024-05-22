import torch
from d2l import torch as d2l
from torch import nn
import datasets
from torchvision import datasets, transforms
from torch.utils import data
import matplotlib.pyplot as plt
import time

cnn_acc = []
dnn_acc = []
dnn_loss = []
cnn_loss = []

batch_size = 64
num_inputs = 3 * 224 * 224
num_outputs = 11
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
test_batch_size = 256

train_augs = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

x = [i for i in range(15)]

train_dataset = datasets.ImageFolder(root='./food/training', transform=train_augs)
test_dataset = datasets.ImageFolder(root="./food/validation", transform=train_augs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

dnn = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, 512), nn.Dropout(0.4), nn.ReLU(), nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 11))


def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 1)


dnn.apply(init_weights)

epochs, lr = 15, 0.0006
loss_fn = nn.CrossEntropyLoss()


def accuracy(lable, target):
    result = (lable.reshape(1, -1) == target)
    return float(result.int().sum()) / float(len(lable))


def correct(lable, target):
    result = (lable.reshape(1, -1) == target)
    return float(result.int().sum())


def train(net, train_iter, step, learning_rate, net_model, device='cuda:0'):
    net.to(device)
    start = time.time()
    trainer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print(f"{net_model} train begin on device: {device}")
    for epoch in range(epochs):
        for (images, labels) in train_iter:
            images, labels = images.to(device), labels.to(device)
            net.zero_grad()
            outputs = net(images)
            outputs.to(device)
            loss = nn.CrossEntropyLoss().to(device)(outputs, labels)
            loss.sum().backward()
            lable = outputs.argmax(dim=1, keepdim=True)
            acc = accuracy(lable, labels)
            trainer.step()
        if net_model == 'dnn':
            dnn_loss.append(loss.item()), dnn_acc.append(acc)
        else:
            cnn_loss.append(loss.item()), cnn_acc.append(acc)
    end = time.time()
    print(f'spend_time: {end - start}s')


train(dnn, train_loader, epochs, lr, net_model='dnn')
cnn = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(16, 16, kernel_size=3), nn.ReLU(),
    nn.Dropout(0.4),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(10816, 512), nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 128), nn.ReLU(),
    nn.Linear(128, 11)
)

train(cnn, test_loader, epochs, lr, net_model='cnn')

plt.plot(x, dnn_loss, marker='o', label='dnn')
plt.title("Adam")
plt.xlabel("step")
plt.ylabel("DNN LOSS")
plt.show()
plt.cla()
plt.plot(x, cnn_loss, marker='o', label='cnn')
plt.xlabel("step")
plt.title("Adam")
plt.ylabel("CNN LOSS")
plt.show()
plt.cla()
plt.plot(x, cnn_acc, marker='o', label='cnn')
plt.plot(x, dnn_acc, marker='o', label='dnn')
plt.xlabel("step")
plt.title("Adam")
plt.ylabel("ACCURACY")
plt.legend()
plt.show()

# torch.save(cnn, 'cnn_file')
# torch.save(dnn,'dnn_file')
# cnn=torch.load('./cnn_file')
# dnn=torch.load('./dnn_file')
# 用于保存和加载模型参数，以避免重复训练而浪费时间


sum = 0

for x, y in test_loader:
    x, y = x.to('cuda'), y.to('cuda')
    dnn.to('cuda')
    dnn.eval()
    output = dnn(x).to('cuda')
    output = output.to('cuda')
    lable = output.argmax(dim=1, keepdim=True)
    sum += correct(lable, y)

print(f'DNN Accuracy: {float(sum) / float(3430)}')

sum = 0

for x, y in test_loader:
    cnn.eval()
    cnn.to('cuda')
    x, y = x.to('cuda'), y.to('cuda')
    output = cnn(x)
    output = output.to('cuda')
    lable = output.argmax(dim=1, keepdim=True)
    sum += correct(lable, y)

print(f'CNN Accuracy: {float(sum) / float(3430)}')
