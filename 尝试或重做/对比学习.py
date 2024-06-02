import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

matplotlib.use('Qt5Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
loss_list = []

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

train_data = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='../data', train=False, transform=transform, download=True)
epochs = 40
learning_rate = 7e-4

def get_data(dataset):
    x_data, y_data = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        x_data.append(x)
        y_data.append(y)
    return x_data, y_data


class my_dataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        possibility = np.random.rand()
        idx1 = np.random.randint(0, len(self.y_data) - 1)
        idx2 = np.random.randint(0, len(self.y_data) - 1)
        while self.y_data[idx1] != y:
            idx1 = np.random.randint(0, len(self.y_data) - 1)
        while self.y_data[idx2] == y:
            idx2 = np.random.randint(0, len(self.y_data) - 1)
        return x, self.x_data[idx1],self.x_data[idx2],y


train_x_data, train_y_data = get_data(train_data)
test_x_data, test_y_data = get_data(test_data)
train_dataset = my_dataset(train_x_data, train_y_data)
test_dataset = my_dataset(test_x_data, test_y_data)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class Siamese(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1568, out_features=128),
        )

    def forward(self, x):
        x = self.features(x)
        return x


def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = F.pairwise_distance(anchor, positive, p=2)
    distance_negative = F.pairwise_distance(anchor, negative, p=2)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()



def train(model, train_loader, optimizer, epochs, device):
    model = model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for i, (x1, x2, x3, y) in enumerate(train_loader):
            x1, x2, x3,y = x1.to(device), x2.to(device), x3.to(device),y.to(device)
            pred1 = model(x1)
            pred2 = model(x2)
            pred3 = model(x3)
            loss = triplet_loss(pred1, pred2, pred3,margin=10.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().numpy()
        loss_list.append(total_loss)
        print(f'epoch:{epoch + 1} total_loss:{total_loss}')

def eval(model, device, test_loader, epochs):
    model = model.to(device)
    model.eval()
    x = [i for i in range(epochs)]
    plt.plot(x, loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    encoded_features = []
    original_labels = []
    with torch.no_grad():
        for data in test_loader:
            anchor_imgs, _, _, labels = data
            anchor_imgs = anchor_imgs.to(device)
            anchor_outputs = model(anchor_imgs)

            encoded_features.extend(anchor_outputs.cpu().numpy())
            original_labels.extend(labels.cpu().numpy())

            if len(encoded_features) >= 7000:
                break
    encoded_features = np.array(encoded_features)
    tsne = TSNE(n_components=2, random_state=42)
    encoded_features_tsne = tsne.fit_transform(encoded_features)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示的字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(12, 10))  # 调整图表大小
    plt.scatter(encoded_features_tsne[:, 0], encoded_features_tsne[:, 1], c=original_labels, cmap=plt.cm.tab10,
                marker='o', alpha=0.7)  # 调整颜色映射和透明度，使用原始标签
    plt.colorbar(ticks=range(10))
    plt.title('孪生网络编码特征的 t-SNE 可视化')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.grid(True)  # 添加网格线
    plt.show()


print('===========Siamese===========')


model = Siamese(output_size=10)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train(model, train_loader, optimizer, epochs, device)
eval(model, device, test_loader, epochs)

class AutoEDcoder(nn.Module):
    def __init__(self):
        super(AutoEDcoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),
            nn.Conv2d(3, 9, 3, 2, 1),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(),
            nn.Conv2d(9, 18, 3, 2, 1),
            nn.BatchNorm2d(18),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(18 * 4 * 4, 128)
        )

    def forward(self, x):
        # Encoder
        code = self.encoder(x)
        return code

loss_list = []

net = AutoEDcoder()
print('===========AutoEDcoder===========')
trainer = torch.optim.Adam(net.parameters(), lr=learning_rate)

train(net, train_loader, trainer, epochs, device)

eval(net, device, test_loader, epochs)
