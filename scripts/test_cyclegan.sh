set -ex
python test.py --dataroot ../datasets/lsb_un --results_dir ./results/ --name lsb_cyclegan --model cycle_gan --netG resnet_9blocks --num_test 50 --phase test --no_dropout --norm instance --expand_bits
