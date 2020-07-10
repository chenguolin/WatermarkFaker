# How to Run the Codes

## 注意：原始项目中尚未包含`dataset`文件夹，需自行添加数据集

## 0x00 将当前路径设置为本文件夹并安装（更新）相应的库
打开一个终端，输入：
```
cd <你存放本文件夹>/GAN-for-LSB-Watermarks
pip install -r requirements.txt
```

## 0x01 利用Visdom可视化
打开一个新的终端，输入：
```
python -m visdom.server
```
然后打开[http://localhost:8097/](http://localhost:8097/)

## 0x02 训练网络：
回到最开始的终端，输入：
- Linux
```
bash ./scripts/train_pix2pix.sh
```
- Windows
```
python train.py --dataroot ./datasets/lsb --name lsb_pix2pix --model pix2pix --netG unet_256 -netD basic --max_dataset_size 12000 -norm batch --n_epochs 5 --n_epochs_decay 5 -gan_mode lsgan --pool_size 0
```

## 0x03 测试网络：
- Linux
```
bash ./scripts/test_pix2pix.sh
```
- Windows
```
python test.py --dataroot ./datasets/lsb --name lsb_pix2pix --model pix2pix --netG unet_256
```