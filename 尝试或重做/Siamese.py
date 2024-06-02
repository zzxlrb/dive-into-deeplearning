import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

# 创建孪生网络的数据集类
class SiameseMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.pairs = self.create_pairs(dataset)

    def create_pairs(self, dataset):
        pairs = []
        labels = dataset.targets
        for i in range(len(dataset)):
            img1, label1 = dataset[i]
            positive_idx = np.random.choice(np.where(labels.numpy() == label1)[0])
            negative_idx = np.random.choice(np.where(labels.numpy() != label1)[0])
            img2, _ = dataset[positive_idx]
            img3, _ = dataset[negative_idx]
            pairs.append((img1, img2, img3, label1))  # 修改此处，返回图像和标签
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img1, img2, img3, label1 = self.pairs[index]  # 修改此处，返回图像和标签
        return img1, img2, img3, label1  # 修改此处，返回图像和标签


# 构建孪生网络模型
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(784, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_features=2)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

# 定义孪生网络的损失函数
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        distance_negative = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        loss = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(loss)

# 训练参数设置
batch_size = 128
num_epochs = 15
learning_rate = 0.001
margin = 1.0

# 准备数据加载器
train_loader = DataLoader(SiameseMNISTDataset(train_dataset), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(SiameseMNISTDataset(test_dataset), batch_size=batch_size, shuffle=False)

# 初始化网络和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
siamese_net = SiameseNetwork().to(device)
criterion = TripletLoss(margin)
optimizer = optim.Adam(siamese_net.parameters(), lr=learning_rate)

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # 训练阶段
    siamese_net.train()
    running_train_loss = 0.0
    correct_train_pairs = 0
    total_train_pairs = 0

    for img1, img2, img3, _ in train_loader:
        img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
        optimizer.zero_grad()
        output1, output2, output3 = siamese_net(img1), siamese_net(img2), siamese_net(img3)
        loss = criterion(output1, output2, output3)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

    # 测试阶段
    siamese_net.eval()
    running_test_loss = 0.0
    correct_test_pairs = 0
    total_test_pairs = 0

    with torch.no_grad():
        for img1, img2, img3, _ in test_loader:
            img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
            output1, output2, output3 = siamese_net(img1), siamese_net(img2), siamese_net(img3)
            loss = criterion(output1, output2, output3)
            running_test_loss += loss.item()
    test_loss = running_test_loss / len(test_loader)
    test_losses.append(test_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, ")

print('训练完成')


# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 获取孪生网络的编码特征
siamese_net.eval()
encoded_features = []  # 用于保存编码后的特征
original_labels = []  # 用于保存原始标签
with torch.no_grad():
    for data in test_loader:
        anchor_imgs, _, _, labels = data
        anchor_imgs = anchor_imgs.to(device)

        # 获取孪生网络的编码输出
        anchor_outputs = siamese_net(anchor_imgs)

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
plt.title('孪生网络编码特征的 t-SNE 可视化')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.grid(True)  # 添加网格线
plt.show()