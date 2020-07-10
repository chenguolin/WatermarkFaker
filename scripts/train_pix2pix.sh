set -ex
python train.py --dataroot ./datasets/lsb --name lsb_pix2pix --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12000 --norm batch --n_epochs 5 --n_epochs_decay 5 --gan_mode lsgan --pool_size 0
