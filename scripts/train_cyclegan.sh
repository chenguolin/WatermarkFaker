set -ex
python train.py --dataroot ../datasets/lsb_un --name lsb_cyclegan --model cycle_gan --pool_size 50 --no_dropout --netG resnet_9blocks --netD basic --max_dataset_size 12000 --norm instance --n_epochs 1 --n_epochs_decay 0 --gan_mode lsgan --display_freq 100 --expand_bits
