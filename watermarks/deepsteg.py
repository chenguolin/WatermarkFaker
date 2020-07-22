import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from utils import util
from watermarks.base_watermark import BaseWatermark

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):
    def __init__(self, root, subfolder, transform):
        super(MyDataset, self).__init__()
        self.path = os.path.join(root, subfolder)
        self.image_list = [_ for _ in os.listdir(self.path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image_path = os.path.join(self.path, self.image_list[item])
        watermark_path = os.path.join(self.path, self.image_list[np.random.randint(len(self.image_list))])
        image = np.array(Image.open(image_path).convert('RGB'))
        watermark = np.array(Image.open(watermark_path).convert('RGB'))
        return {'image': self.transform(image), 'watermark': self.transform(watermark)}
        

def loadData(root, subfolder, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    dataset = MyDataset(root, subfolder, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # preparation network
        self.conv1 = nn.Conv2d( 3, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d( 3, 10, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d( 3,  5, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv6 = nn.Conv2d(65,  5, kernel_size=5, padding=2)
        # hidden network
        self.conv7  = nn.Conv2d(68, 50, kernel_size=3, padding=1)
        self.conv8  = nn.Conv2d(68, 10, kernel_size=4, padding=1)
        self.conv9  = nn.Conv2d(68,  5, kernel_size=5, padding=2)
        self.conv10 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv12 = nn.Conv2d(65,  5, kernel_size=5, padding=2)
        self.conv13 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv15 = nn.Conv2d(65,  5, kernel_size=5, padding=2)
        self.conv16 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv17 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv18 = nn.Conv2d(65,  5, kernel_size=5, padding=2)
        self.conv19 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv20 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv21 = nn.Conv2d(65,  5, kernel_size=5, padding=2)
        self.conv22 = nn.Conv2d(65,  3, kernel_size=3, padding=1)

    def forward(self, image, watermark):
        # prepration
        x1 = F.relu(self.conv1(watermark))
        x2 = F.relu(self.conv2(watermark))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv3(watermark))
        x4 = torch.cat([x1, x2, x3], dim=1)

        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv6(x4))            
        x4 = torch.cat([x1, x2, x3], dim=1)

        # hidden
        x4 = torch.cat([image, x4], dim=1)
        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat([x1, x2, x3], dim=1)

        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat([x1, x2, x3], dim=1)

        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat([x1, x2, x3], dim=1)

        x1 = F.relu(self.conv16(x4))
        x2 = F.relu(self.conv17(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv18(x4))
        x4 = torch.cat([x1, x2, x3], dim=1)

        x1 = F.relu(self.conv19(x4))
        x2 = F.relu(self.conv20(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv21(x4))
        x4 = torch.cat([x1, x2, x3], dim=1)

        image_wm =F.relu(self.conv22(x4))
        return image_wm


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # decoder
        self.conv1  = nn.Conv2d( 3, 50, kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d( 3, 10, kernel_size=4, padding=1)
        self.conv3  = nn.Conv2d( 3,  5, kernel_size=5, padding=2)
        self.conv4  = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv5  = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv6  = nn.Conv2d(65,  5, kernel_size=5, padding=2)
        self.conv7  = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv8  = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv9  = nn.Conv2d(65,  5, kernel_size=5, padding=2)
        self.conv10 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv12 = nn.Conv2d(65,  5, kernel_size=5, padding=2)
        self.conv13 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv15 = nn.Conv2d(65,  5, kernel_size=5, padding=2)
        self.conv16 = nn.Conv2d(65,  3, kernel_size=3, padding=1)

    def addGaussianNoise(self, x, std=0.1):
        assert isinstance(x, torch.Tensor)
        return x + (torch.randn(x.shape) * (std ** 2)).to(device)

    def forward(self, image_wm):
        x = self.addGaussianNoise(image_wm, std=0.1)
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv3(x))
        x4 = torch.cat([x1, x2, x3], dim=1)

        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv6(x4))            
        x4 = torch.cat([x1, x2, x3], dim=1)

        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat([x1, x2, x3], dim=1)

        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat([x1, x2, x3], dim=1)

        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', 0)
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat([x1, x2, x3], dim=1)

        watermark_ = F.relu(self.conv16(x4))
        return watermark_


class DeepStego(nn.Module):
    def __init__(self):
        super(DeepStego, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, image, watermark):
        image_wm = self.encoder(image, watermark)
        watermark_ = self.decoder(image_wm)
        return image_wm, watermark_


def train(continue_train=False):
    epochs = 10
    learning_rate = 0.001
    batch_size = 1
    beta = 1.0

    model = DeepStego().to(device)
    if continue_train:
        model.load_state_dict(torch.load("./watermarks/deepsteg.pth"))
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = loadData("../datasets/lsb_un", "trainA", batch_size, shuffle=False)
    data_size = len(train_loader)
    x, train_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, dic in enumerate(train_loader):
            optimizer.zero_grad()
            image, watermark = dic['image'].to(device), dic['watermark'].to(device)
            image_wm, watermark_ = model.forward(image, watermark)

            encoder_loss = criterion(image_wm.view((-1, 256*256*3)), image.view((-1, 256*256*3)))
            decoder_loss = criterion(watermark_.view((-1, 256*256*3)), watermark.view((-1, 256*256*3)))
            loss = encoder_loss + beta * decoder_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 200 == 0:
                print("[%03d]  Train loss in batch [%05d]: %f" % (epoch+1, i+1, loss.item()))
                util.save_image(util.tensor2im(image), "./temp/image.png")
                util.save_image(util.tensor2im(watermark), "./temp/watermark.png")
                util.save_image(util.tensor2im(image_wm), "./temp/encode_out.png")
                util.save_image(util.tensor2im(watermark_), "./temp/decoder_out.png")
            if i % 4000 == 0:
                x.append(epoch + i/data_size)
                train_losses.append(loss.item())
                plt.plot(x, train_losses)
                plt.savefig("./temp/loss.png")
                plt.clf()
        print("Average train loss in epoch [%03d]: %f" % (epoch+1, train_loss/data_size))
        torch.save(model.state_dict(), "./watermarks/deepsteg.pth")
