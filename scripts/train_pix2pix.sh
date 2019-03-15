set -ex
python train.py --dataroot ./datasets/flags --name flags_pix2pix --model pix2pix --netG unet_128 --input_nc 9 --direction AtoB --lambda_L1 100 --dataset_mode unaligned_hdf5 --norm batch --pool_size 0 --gpu_ids -1
