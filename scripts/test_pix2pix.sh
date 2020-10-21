set -ex
python test.py --dataroot ../datasets/lsb --results_dir ./results/ --name lsb-watermark --model pix2pix --netG unet_256 --num_test 50 --phase test --eval --expand_bits --epoch latest
