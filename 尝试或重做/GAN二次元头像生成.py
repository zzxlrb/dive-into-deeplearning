import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import os
import matplotlib.image as mpimg


class Config(object):
    data_dir = '../data/faces'
    img_size = 96
    batch_size = 256
    epochs = 2
    glr = 2e-4
    dlr = 2e-4
    betal = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nz = 100
    ngf = 64  # 生成器的卷积核个数
    ndf = 64  # 判别器的卷积核个数
    save_path = "./模型参数"
    d_every = 1
    g_every = 1
    gen_img = "./result/result_"
    # 选择保存的照片的位置
    # 一次生成保存64张图片
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 生成模型的噪声均值
    gen_std = 1  # 噪声方差
    g_loss_list,d_loss_list=[],[]
    path="./result"


class Generator(nn.Module):
    def __init__(self, opt: Config):
        super(Generator, self).__init__()
        self.ngf = opt.ngf
        self.generate = nn.Sequential(
            nn.ConvTranspose2d(in_channels=opt.nz, out_channels=self.ngf * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.ngf * 8, out_channels=self.ngf * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.ngf * 4, out_channels=self.ngf * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, X):
        return self.generate(X)


class Discriminator(nn.Module):
    def __init__(self, opt: Config):
        super(Discriminator, self).__init__()
        self.ndf = opt.ndf
        self.Discrim = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.ndf * 4, out_channels=self.ndf * 8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.Discrim(x).view(-1)


@torch.no_grad()
def eval(opt: Config, netG, netD,epoch):
    device = opt.device
    netG.eval()
    netD.eval()
    netG.to(device)
    netD.to(device)
    noise = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std).to(device)
    fake_image = netG(noise)
    score = netD(fake_image).detach()
    index = score.topk(opt.gen_num)[1]
    result = []
    for i in index:
        result.append(fake_image.data[i])
    # 以opt.gen_img为文件名保存生成图片
    torchvision.utils.save_image(torch.stack(result), opt.gen_img+str(epoch)+".png", normalize=True)


def train(opt: Config):
    transformers = torchvision.transforms.Compose([
        torchvision.transforms.Resize((opt.img_size, opt.img_size)),
        torchvision.transforms.CenterCrop(opt.img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    device = opt.device
    dataset = torchvision.datasets.ImageFolder(root=opt.data_dir, transform=transformers)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,drop_last=True)
    netG, netD = Generator(opt), Discriminator(opt)
    netG.to(opt.device)
    netD.to(opt.device)
    netG.train()
    netD.train()
    optimize_g = torch.optim.Adam(netG.parameters(), lr=opt.glr, betas=(opt.betal, 0.999))
    optimize_d = torch.optim.Adam(netD.parameters(), lr=opt.dlr, betas=(opt.betal, 0.999))
    criterion = nn.BCELoss().to(device)
    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)
    noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=opt.device)
    for epoch in tqdm(range(opt.epochs)):
        total_g_loss,total_d_loss=0,0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            if i % opt.d_every == 0:
                optimize_d.zero_grad()
                output = netD(images)
                loss = criterion(output, true_labels)
                loss.backward()
                noise = noise.detach()
                fake_images = netG(noise).detach()
                output = netD(fake_images)
                loss_d = criterion(output, fake_labels)
                loss_d.backward()
                optimize_d.step()
                total_d_loss+=loss_d.item()+loss.item()
            if i % opt.g_every == 0:
                optimize_g.zero_grad()
                noise.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1, device=opt.device))
                fake_images = netG(noise)
                output = netD(fake_images)
                loss_g = criterion(output, true_labels)
                loss_g.backward()
                optimize_g.step()
                total_g_loss+=loss_g.item()
        opt.d_loss_list.append(total_d_loss)
        opt.g_loss_list.append(total_g_loss)
        if (epoch+1) % 1 == 0:
            if not os.path.exists(opt.path):
                os.makedirs(opt.path)
            eval(opt,netG,netD,epoch+1)
    return netG, netD

opt = Config()
netG, netD = train(opt)
# torch.save(netG,'./模型参数/netG.pth')
# torch.save(netD,'./模型参数/netD.pth')
# netG = torch.load('./模型参数/netG.pth')
# netD = torch.load('./模型参数/netD.pth')

length = len(opt.d_loss_list)
x=[i for i in range(length)]
plt.plot(x, opt.g_loss_list,label='G_loss')
plt.plot(x, opt.d_loss_list,label='D_loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
for png_file_path in os.listdir(opt.path):
    I = mpimg.imread(opt.path+"/" + png_file_path)
    plt.imshow(I)
    plt.title("epoch: "+str(png_file_path.split('_')[1].split('.')[0]))
    plt.show()