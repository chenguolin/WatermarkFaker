set -ex
python train.py --dataroot ../datasets/lsb --name lsb_pix2pixbits --model pix2pixbits --netG unet_256 --netD basic --max_dataset_size 12000 --norm batch --n_epochs 3 --n_epochs_decay 3 --gan_mode vanilla --pool_size 0 -- display_freq 100
