import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from utils import util

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
        watermark_path = os.path.join(self.path, self.image_list[(item+1)%len(self.image_list)])
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


class StegNet(nn.Module):
    def __init__(self):
        super(StegNet, self).__init__()
        self.define_encoder()
        self.define_decoder()

    def define_encoder(self):
        # layer1
        self.encoder_payload_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.encoder_source_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # layer2
        self.encoder_payload_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.encoder_source_21 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.encoder_bn2 = nn.BatchNorm2d(32)
        # layer3
        self.encoder_payload_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # layer4
        self.encoder_payload_4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.encoder_source_41 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.encoder_bn4 = nn.BatchNorm2d(32)
        # layer5
        self.encoder_payload_5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # layer6
        self.encoder_payload_6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_6 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.encoder_source_61 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.encoder_source_62 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.encoder_bn6 = nn.BatchNorm2d(32)
        # layer7
        self.encoder_payload_7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # layer8
        self.encoder_payload_8 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.encoder_source_81 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.encoder_source_82 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.encoder_bn8 = nn.BatchNorm2d(32)
        # layer9
        self.encoder_source_9 = nn.Conv2d(32, 16, kernel_size=1)
        # layer10
        self.encoder_source_10 = nn.Conv2d(16, 8, kernel_size=1)
        # layer11
        self.encoder_source_11 = nn.Conv2d(8, 3, kernel_size=1)

    def define_decoder(self):
        self.decoder_layers1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.decoder_layers2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # self.decoder_bn2 = nn.BatchNorm2d(64)
        self.decoder_layers3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_layers4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.decoder_bn4 = nn.BatchNorm2d(32)
        self.decoder_layers5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # payload_decoder
        self.decoder_payload1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.decoder_payload2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.decoder_payload3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.decoder_payload4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.decoder_payload5 = nn.Conv2d(8, 3, kernel_size=3, padding=1)
        self.decoder_payload6 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        # source_decoder
        self.decoder_source1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.decoder_source2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.decoder_source3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.decoder_source4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.decoder_source5 = nn.Conv2d(8, 3, kernel_size=3, padding=1)
        self.decoder_source6 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        source, payload = x
        s = source.view((-1, 3, 256, 256))
        p = payload.view((-1, 1, 256, 256))
        
        # ------------------- Encoder ------------------- #
        # layer1
        p = F.relu(self.encoder_payload_1(p))
        s = F.relu(self.encoder_source_1(s))
        #layer2
        p = F.relu(self.encoder_payload_2(p))
        s1 = torch.cat((s, p), 1)  # 64
        s = F.relu(self.encoder_source_2(s1))
        s = F.relu(self.encoder_source_21(s1))
        # s = self.encoder_bn2(s)
        # layer3
        p = F.relu(self.encoder_payload_3(p))
        s = F.relu(self.encoder_source_3(s))
        # layer4
        p = F.relu(self.encoder_payload_4(p))
        s2 = torch.cat((s, p, s1), 1)  # 128
        s = F.relu(self.encoder_source_4(s2))
        s = F.relu(self.encoder_source_41(s))
        # s = self.encoder_bn4(s)
        # layer5
        p = F.relu(self.encoder_payload_5(p))
        s = F.relu(self.encoder_source_5(s))
        # layer6
        p = F.relu(self.encoder_payload_6(p))
        s3 = torch.cat((s, p, s2), 1) # 192
        s = F.relu(self.encoder_source_6(s3))
        s = F.relu(self.encoder_source_61(s))
        s = F.relu(self.encoder_source_62(s))
        # s = self.encoder_bn6(s)
        # layer7
        p = F.relu(self.encoder_payload_7(p))
        s = F.relu(self.encoder_source_7(s))
        # layer8
        p = F.relu(self.encoder_payload_8(p))
        s4 = torch.cat((s, p, s3), 1)
        s = F.relu(self.encoder_source_8(s4))
        s = F.relu(self.encoder_source_81(s))
        s = F.relu(self.encoder_source_82(s))
        # s = self.encoder_bn8(s)
        # layer9
        s = F.relu(self.encoder_source_9(s))
        # layer10
        s = F.relu(self.encoder_source_10(s))
        # layer11
        encoder_output = self.encoder_source_11(s)
        
        # ------------------- Decoder ------------------- #
        d = encoder_output.view(-1, 3, 256, 256)
        
        # layer1
        d = F.relu(self.decoder_layers1(d))
        d = F.relu(self.decoder_layers2(d))
        # d = self.decoder_bn2(d)
        # layer3
        d = F.relu(self.decoder_layers3(d))
        d = F.relu(self.decoder_layers4(d))
        # d = self.decoder_bn4(d)
        init_d = F.relu(self.decoder_layers5(d))

        # ------------------- decoder_payload ------------------- #
        # layer 1 & 2
        d = F.relu(self.decoder_payload1(init_d))
        d = F.relu(self.decoder_payload2(d))
        # layer 3 & 4
        d = F.relu(self.decoder_payload3(d))
        d = F.relu(self.decoder_payload4(d))
        # layer 5 & 6
        d = F.relu(self.decoder_payload5(d))
        decoded_payload = self.decoder_payload6(d)
        
        # ------------------- decoder_source ------------------- #
        # layer 1 & 2
        d = F.relu(self.decoder_source1(init_d))
        d = F.relu(self.decoder_source2(d))
        # layer 3 & 4
        d = F.relu(self.decoder_source3(d))
        d = F.relu(self.decoder_source4(d))
        # layer 5 & 6
        d = F.relu(self.decoder_source5(d))
        decoded_source = self.decoder_source6(d)
        
        return encoder_output, decoded_payload, decoded_source


def train():
    epochs = 10
    train_losses = []
    batch_size = 1

    model = StegNet().to(device)
    criterion = nn.MSELoss().to(device)
    metric = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = loadData("../datasets/lsb_un", "trainA", batch_size, shuffle=False)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, dic in enumerate(train_loader):
            optimizer.zero_grad()
            image, watermark = dic['image'].to(device), dic['watermark'].to(device)
            e_out, dp_out, ds_out = model.forward((image, watermark))

            e_loss = criterion(e_out.view((-1, 256*256*3)), image.view((-1, 256*256*3)))
            dp_loss = criterion(dp_out.view((-1, 256*256)), watermark.view((-1, 256*256)))
            ds_loss = criterion(ds_out.view((-1, 256*256*3)), image.view((-1, 256*256*3)))
            loss = e_loss + dp_loss + ds_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 200 == 0:
                print("[%03d]  Train loss in batch [%05d]: %f" % (epoch+1, i+1, loss.item()))
                util.save_image(util.tensor2im(image), "./temp/image.png")
                util.save_image(util.tensor2im(watermark), "./temp/watermark.png")
                util.save_image(util.tensor2im(e_out), "./temp/encode_out.png")
                util.save_image(util.tensor2im(dp_out), "./temp/decoder_payload.png")
                util.save_image(util.tensor2im(ds_out), "./temp/decoder_source.png")
        print("Average train loss in epoch [%03d]: %f" % (epoch+1, train_loss/len(train_loader)))
        torch.save(model.state_dict(), "./watermarks/cnned.pth")
