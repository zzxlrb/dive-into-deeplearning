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
            nn.Conv2d(16, 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(3136, out_features=4096),
        )

    def forward(self, x):
        x = self.features(x)
        return x


def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = F.pairwise_distance(anchor, positive, p=2)
    distance_negative = F.pairwise_distance(anchor, negative, p=2)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()


model = Siamese(output_size=10)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


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
    tsne3d = TSNE(n_components=3, init='pca', perplexity=30., random_state=0, learning_rate=300)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    with torch.no_grad():
        num = 0
        for x, _, _, y in test_loader:
            x, y = x.to(device), y.detach().cpu().numpy()
            y_pred = model(x)
            np_y_pred = y_pred.detach().cpu().numpy()
            result = tsne3d.fit_transform(np_y_pred)
            x_min, x_max = result.min(0), result.max(0)
            result = (result - x_min) / (x_max - x_min)
            ax.scatter(result[:, 0], result[:, 1], result[:, 2], c=y, cmap='jet')
            for i, txt in enumerate(y):
                ax.text(result[i, 0], result[i, 1], result[i, 2], str(txt))
            num += 1
            if num == 2:
                break
        plt.show()
    tsne2d = TSNE(n_components=2, init='pca', perplexity=30., random_state=0, learning_rate=300)
    with (torch.no_grad()):
        num = 0
        for x, _, _, y in test_loader:
            x, y = x.to(device), y.detach().cpu().numpy()
            y_pred = model(x)
            np_y_pred = y_pred.detach().cpu().numpy()
            result = tsne2d.fit_transform(np_y_pred)
            x_min, x_max = result.min(0), result.max(0)
            result = (result - x_min) / (x_max - x_min)
            plt.scatter(result[:, 0], result[:, 1], c=y, cmap='jet')
            num += 1
            if num == 2:
                break
        plt.show()


epochs = 20


train(model, train_loader, optimizer, epochs, device)

torch.save(model, 'Siamese.pth')

# model = torch.load('Siamese.pth')
#
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
            nn.Linear(18 * 4 * 4, 512),
            nn.Linear(512,1024)
        )

    def forward(self, x):
        # Encoder
        code = self.encoder(x)
        return code

loss_list = []

net = AutoEDcoder()
trainer = torch.optim.Adam(net.parameters(), lr=5e-4)

train(net, train_loader, trainer, epochs, device)

eval(net, device, test_loader, epochs)
