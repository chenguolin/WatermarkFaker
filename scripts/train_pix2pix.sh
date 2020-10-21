set -ex
python train.py --dataroot ../datasets/lsb --name lsb-watermark --model pix2pix --netG unet_256 --netD basic --norm batch --n_epochs 10 --n_epochs_decay 0 --gan_mode lsgan --lambda_L1 10.0 --display_freq 100 --expand_bits --input_nc 3 --output_nc 3
