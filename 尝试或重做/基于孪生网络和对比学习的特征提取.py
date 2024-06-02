import matplotlib
from d2l import torch as d2l
import torch
from sklearn.manifold import TSNE
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Qt5Agg')

batch_size = 128

loss_list = []

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

train_data = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

size = 1 * 28 * 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            nn.Flatten()
        )
        self.fc1 = nn.Linear(18 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 18 * 7 * 7)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(18, 9, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(9, 1, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        code = self.fc1(x)
        # Decoder
        x = self.fc2(code)
        x = x.view(x.size(0), 18, 7, 7)
        out = self.decoder(x)
        return out


def train(model, device, train_loader, optimizer, epochs):
    model = model.to(device)
    model.train(True)
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = nn.MSELoss()(x, y_pred)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'epoch:{epoch}  total_loss:{total_loss}')
        loss_list.append(total_loss)


def eval(model, device, test_loader, epochs):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            y_pred = model(x)
            # 可视化重构结果和原始图像
            fig, axes = plt.subplots(16, 2, figsize=(x[0].size(0) * 2, x[0].size(1) / 1.5))
            axes[0, 0].set_title('Original')
            axes[0, 1].set_title('Reconstructed')
            # 原始图像
            for i in range(16):
                # 原始图像
                axes[i, 0].imshow(x[i].cpu().squeeze(), cmap='gray')
                axes[i, 0].axis('off')

                # 重构图像
                axes[i, 1].imshow(y_pred[i].cpu().squeeze(), cmap='gray')
                axes[i, 1].axis('off')
            plt.tight_layout()
            plt.show()
            break
    x = [i for i in range(epochs)]
    plt.plot(x, loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    model.eval()
    encoded_features = []  # 用于保存编码后的特征
    original_labels = []  # 用于保存原始标签
    with torch.no_grad():
        for data in test_loader:
            anchor_imgs,labels = data
            anchor_imgs = anchor_imgs.to(device)

            # 获取孪生网络的编码输出
            anchor_outputs = model.encoder(anchor_imgs)
            # 将编码输出转换为 NumPy 数组并添加到列表中
            encoded_features.extend(anchor_outputs.cpu().numpy())

            # 添加对应的原始标签
            original_labels.extend(labels.cpu().numpy())

            if len(encoded_features) >= 7000:
                break

    # 将列表转换为 NumPy 数组
    encoded_features = np.array(encoded_features)

    # 执行 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    encoded_features_tsne = tsne.fit_transform(encoded_features)

    # 绘制 t-SNE 可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示的字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(12, 10))  # 调整图表大小
    plt.scatter(encoded_features_tsne[:, 0], encoded_features_tsne[:, 1], c=original_labels, cmap=plt.cm.tab10,
                marker='o', alpha=0.7)  # 调整颜色映射和透明度，使用原始标签
    plt.colorbar(ticks=range(10))
    plt.title('自动编码器的 t-SNE 可视化')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.grid(True)  # 添加网格线
    plt.show()


net = AutoEDcoder()

epochs = 20

train(model=net, device=device, train_loader=train_loader, optimizer=torch.optim.Adam(net.parameters(), lr=0.3),
      epochs=epochs)

net = torch.load('AutoEDcoder.pth')
eval(net, device=device, test_loader=test_loader, epochs=epochs)