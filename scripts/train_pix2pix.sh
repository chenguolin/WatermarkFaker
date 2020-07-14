set -ex
python train.py --dataroot ../datasets/lsb --name lsb_pix2pix-bits --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12000 --norm batch --n_epochs 1 --n_epochs_decay 0 --gan_mode vanilla --pool_size 0 --display_freq 100 --expand_bits
