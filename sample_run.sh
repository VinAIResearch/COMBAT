#!/bin/bash

# Train the generator and a surrogate classifier. The posisoning rate on target-label images is 0.5. It is equivalent to ~5% of the entire training set with 10 equal-size classes (0.5 * 100%/10)
python train_cleanlabel_generator.py --dataset cifar10 --pc 0.5 --saving_prefix model1
# Another tab: tensorboard --logdir=checkpoints/model1_clean/cifar10/log_dir

# Train a classifier with clean-label poisoned data using the trained generator. Here we randomize the image to pison and fix them during training
python verify_cleanlabel_generator_fixed.py --dataset cifar10 --pc 0.5 --saving_prefix verify_model1 --load_checkpoint model1_clean





