set -ex
python train.py --dataroot ../datasets/lsb --name pix2pix-expand-lsb --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 10 --n_epochs_decay 0 --gan_mode vanilla --pool_size 0 --display_freq 100 --expand_bits --watermark lsb --save_epoch_freq 1
