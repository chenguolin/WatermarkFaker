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
        watermark = np.array(Image.open(watermark_path).convert('L'))
        return {'image': self.transform(image), 'watermark': self.transform(watermark)}
        


def loadData(root, subfolder, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    dataset = MyDataset(root, subfolder, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class StegNet(nn.Module, BaseWatermark):
    def __init__(self, save_watermark=False):
        super(StegNet, self).__init__()
        self.save_watermark = save_watermark
        # Encoder
        self.encoder_payload_1_16 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # same padding
            nn.ReLU(inplace=True)
        )
        self.encoder_cover_3_16 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder_payload_16_16 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder_cover_16_16 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder_cover_16_8 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder_cover_32_16 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2d_8_3 = nn.Sequential(
            nn.Conv2d(8, 3, kernel_size=3, padding=1)
        )
        # Decoder
        self.decoder_stego_3_16 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder_stego_16_16 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder_stego_16_8 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder_stego_8_8 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder_stego_8_3 = nn.Sequential(
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder_stego_3_3 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2d_3_1 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, padding=1)
        )

    def embed(self, image, watermark):
        """Encoder"""
        if image.shape[1] != 3 or watermark.shape[1] != 1:
            raise TypeError("Image should be RGB batch tensor and watermark should be grayscale batch tensor")
        
        # layer1
        payload = self.encoder_payload_1_16(watermark)
        cover = self.encoder_cover_3_16(image)
        cover = torch.cat((cover, payload), dim=1)
        # layer2
        payload = self.encoder_payload_16_16(payload)
        cover = self.encoder_cover_32_16(cover)
        #layer3
        payload = self.encoder_payload_16_16(payload)
        cover = self.encoder_cover_16_16(cover)
        cover = torch.cat((cover, payload), dim=1)
        # layer4
        payload = self.encoder_payload_16_16(payload)
        cover = self.encoder_cover_32_16(cover)
        # layer5
        payload = self.encoder_payload_16_16(payload)
        cover = self.encoder_cover_16_16(cover)
        cover = torch.cat((cover, payload), dim=1)
        # layer6
        payload = self.encoder_payload_16_16(payload)
        cover = self.encoder_cover_32_16(cover)
        # layer7
        payload = self.encoder_payload_16_16(payload)
        cover = self.encoder_cover_16_16(cover)
        cover = torch.cat((cover, payload), dim=1)
        # layer8
        stego = self.encoder_cover_32_16(cover)
        # layer9
        stego = self.encoder_cover_16_8(stego)
        # layer10
        image_wm = self.conv2d_8_3(stego)
        return image_wm

    def extract(self, image_wm, image=None):
        """Decoder"""
        if image_wm.shape[1] != 3:
            raise TypeError("Image with watermark should be RGB batch tensor")
        
        # layer1
        stego = self.decoder_stego_3_16(image_wm)
        # layer2
        stego = self.decoder_stego_16_16(stego)
        # layer3
        stego = self.decoder_stego_16_8(stego)
        # layer4
        stego = self.decoder_stego_8_8(stego)
        # layer5
        stego = self.decoder_stego_8_3(stego)
        # layer6
        stego = self.decoder_stego_3_3(stego)
        # layer7
        watermark_ = self.conv2d_3_1(stego)
        return watermark_

    def forward(self, X):
        image, watermark = X
        image_wm = self.embed(image, watermark)
        watermark_ = self.extract(image_wm)
        return image_wm, watermark_


def train():
    epochs = 10
    learning_rate = 0.0001
    batch_size = 1
    alpha, beta = 1, 1

    model = StegNet().to(device)
    criterion = nn.MSELoss().to(device)
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
            image_wm, watermark_ = model.forward((image, watermark))

            encoder_loss = criterion(image_wm.view((-1, 256*256*3)), image.view((-1, 256*256*3)))
            decoder_loss = criterion(watermark_.view((-1, 256*256)), watermark.view((-1, 256*256)))
            loss = alpha * encoder_loss + beta * decoder_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 200 == 0:
                print("[%03d]  Train loss in batch [%05d]: %f" % (epoch+1, i+1, loss.item()))
                x.append(epoch + i/data_size)
                train_losses.append(loss.item())
                util.save_image(util.tensor2im(image), "./temp/image.png")
                util.save_image(util.tensor2im(watermark), "./temp/watermark.png")
                util.save_image(util.tensor2im(image_wm), "./temp/encode_out.png")
                util.save_image(util.tensor2im(watermark_), "./temp/decoder_out.png")
                plt.plot(x, train_losses)
                plt.savefig("./temp/loss.png")
                plt.clf()
        print("Average train loss in epoch [%03d]: %f" % (epoch+1, train_loss/data_size))
        torch.save(model.state_dict(), "./watermarks/cnned.pth")
