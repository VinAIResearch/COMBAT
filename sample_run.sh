#!/bin/bash

## Train the generator and a surrogate classifier. The posisoning rate on target-label images is 0.5. It is equivalent to ~5% of the entire training set with 10 equal-size classes (0.5 * 100%/10)
# python train_cleanlabel_generator.py --dataset cifar10 --noise_rate 0.08 --pc 0.5 --saving_prefix noise_n008
## Another tab: tensorboard --logdir=checkpoints/model1_clean/cifar10/log_dir

## Train a classifier with clean-label poisoned data using the trained generator. Here we randomize the image to pison and fix them during training
# python verify_cleanlabel_generator_fixed.py --dataset cifar10 --noise_rate 0.08 --pc 0.5 --saving_prefix verify_noise_n008 --load_checkpoint noise_n008_clean


# Input-aware
python train_cleanlabel_inputaware_generator.py --dataset cifar10 --noise_rate 0.08 --pc 0.5 --saving_prefix ia_n008
python verify_cleanlabel_inputaware_generator_fixed.py --dataset cifar10 --noise_rate 0.08 --pc 0.5 --saving_prefix verify_ia_n008 --load_checkpoint ia_n008_clean

# Mix noise & warping with 2x2 grid
# python train_cleanlabel_mixed_generator.py --dataset cifar10 --grid_rescale 0.15 --noise_rate 0.05 --s 2 --pc 0.5 --saving_prefix mixed_s2_015_n008
# python verify_cleanlabel_mixed_generator.py --dataset cifar10 --grid_rescale 0.15 --noise_rate 0.05 --s 2 --pc 0.5 --saving_prefix verify_mixed_s2_015_n008 --load_checkpoint mixed_s2_015_n008_clean





