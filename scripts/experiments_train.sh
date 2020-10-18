set -ex

python train.py --dataroot ../datasets/lsb --name lsb-vanilla-nodropout --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 10 --n_epochs_decay 0 --gan_mode vanilla --pool_size 0 --display_freq 100 --watermark lsb --save_epoch_freq 1 --lambda_L1 100.0 --no_dropout --input_nc 3 --output_nc 3
python train.py --dataroot ../datasets/lsb --name lsb-expand-vanilla-nodropout --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 10 --n_epochs_decay 0 --gan_mode vanilla --pool_size 0 --display_freq 100 --expand_bits --watermark lsb --save_epoch_freq 1 --lambda_L1 100.0 --no_dropout --input_nc 3 --output_nc 3
python train.py --dataroot ../datasets/lsb --name lsb-nodropout --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 10 --n_epochs_decay 0 --gan_mode lsgan --pool_size 0 --display_freq 100 --watermark lsb --save_epoch_freq 1 --lambda_L1 10.0 --no_dropout --input_nc 3 --output_nc 3

python train.py --dataroot ../datasets/lsbmr --name lsbmr-expand-vanilla --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 10 --n_epochs_decay 0 --gan_mode vanilla --pool_size 0 --display_freq 100 --expand_bits --watermark lsbmr --save_epoch_freq 1 --lambda_L1 100.0
python train.py --dataroot ../datasets/lsbm --name lsbm-expand-vanilla --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 10 --n_epochs_decay 0 --gan_mode vanilla --pool_size 0 --display_freq 100 --expand_bits --watermark lsbm --save_epoch_freq 1 --lambda_L1 100.0

python train.py --dataroot ../datasets/dct --name dct-vanilla --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 5 --n_epochs_decay 0 --gan_mode vanilla --pool_size 0 --display_freq 100 --watermark dct --save_epoch_freq 1 --lambda_L1 100.0
python train.py --dataroot ../datasets/dct --name dct --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 5 --n_epochs_decay 0 --gan_mode lsgan --pool_size 0 --display_freq 100 --watermark dct --save_epoch_freq 1 --lambda_L1 10.0

python train.py --dataroot ../datasets/lsbmr --name lsbmr-vanilla --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 10 --n_epochs_decay 0 --gan_mode vanilla --pool_size 0 --display_freq 100 --watermark lsbmr --save_epoch_freq 1 --lambda_L1 100.0
python train.py --dataroot ../datasets/lsbmr --name lsbmr --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 10 --n_epochs_decay 0 --gan_mode lsgan --pool_size 0 --display_freq 100 --watermark lsbmr --save_epoch_freq 1 --lambda_L1 10.0

python train.py --dataroot ../datasets/lsbm --name lsbm-vanilla --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 10 --n_epochs_decay 0 --gan_mode vanilla --pool_size 0 --display_freq 100 --watermark lsbm --save_epoch_freq 1 --lambda_L1 100.0
python train.py --dataroot ../datasets/lsbm --name lsbm --model pix2pix --netG unet_256 --netD basic --max_dataset_size 12288 --norm batch --n_epochs 10 --n_epochs_decay 0 --gan_mode lsgan --pool_size 0 --display_freq 100 --watermark lsbm --save_epoch_freq 1 --lambda_L1 10.0
