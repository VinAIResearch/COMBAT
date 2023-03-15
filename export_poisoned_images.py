import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import torchvision.transforms.functional as fn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from torchvision.utils import save_image
from tqdm import tqdm

import config
from classifier_models import PreActResNet10, PreActResNet18, ResNet18
from networks.models import (AE, Denormalizer, NetC_MNIST, NetC_MNIST2,
                             NetC_MNIST3, Normalizer, UnetGenerator)
from utils.dataloader_infer import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from utils.dct import dct_2d, idct_2d

# gauss_smooth = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1))
gauss_smooth = T.GaussianBlur(kernel_size=3, sigma=0.55)


def create_dir(path_dir):
    list_subdir = path_dir.strip(".").split("/")
    list_subdir.remove("")
    base_dir = "./"
    for subdir in list_subdir:
        base_dir = os.path.join(base_dir, subdir)
        try:
            os.mkdir(base_dir)
        except:
            pass


def get_model(opt):
    netG = None

    if opt.dataset == "cifar10":
        netG = UnetGenerator(opt).to(opt.device)
    if opt.dataset == "gtsrb":
        netG = UnetGenerator(opt).to(opt.device)
    if opt.dataset == "mnist":
        netG = UnetGenerator(opt, in_channels=1).to(opt.device)
    if opt.dataset == "celeba":
        netG = UnetGenerator(opt).to(opt.device)
    if opt.dataset in ['imagenet10', 'tinyimagenet']:
        netG = UnetGenerator(opt).to(opt.device)

    # Optimizer
    return netG


def save_images(save_dir, images, ids):
    for image, idx in zip(images, ids):
        filename = f"{idx}.png"
        save_image(image, os.path.join(save_dir, filename))


def low_freq(x, opt):
    image_size = opt.input_height
    ratio = opt.ratio
    mask = torch.zeros_like(x)
    mask[:, :, :int(image_size * ratio), :int(image_size * ratio)] = 1
    x_dct = dct_2d((x+1)/2*255)
    x_dct *= mask
    x_idct = (idct_2d(x_dct)/255*2) - 1
    return x_idct


def make_inputs_bd(netG, inputs, opt):
    # Create backdoor data
    noise_bd = netG(inputs)
    noise_bd = low_freq(noise_bd, opt)
    inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)
    inputs_bd = gauss_smooth(inputs_bd)

    return inputs_bd


def infer(netG, dl, opt, prefix):
    print(" Get poisoned images:")
    cln_save_dir = os.path.join(opt.img_dir, prefix, "cln")
    bd_save_dir = os.path.join(opt.img_dir, prefix, "bd")
    create_dir(cln_save_dir)
    create_dir(bd_save_dir)

    denormalizer = Denormalizer(opt)

    total_imgs = 0
    pbar = tqdm(dl, desc=prefix)
    for inputs, targets, ids in pbar:
        inputs, targets, ids = inputs.to(opt.device), targets.to(opt.device), ids.to(opt.device)

        trg_ind = (targets == opt.target_label).nonzero()[:, 0]
        inputs = inputs[trg_ind]
        ids = ids[trg_ind]

        if len(inputs) == 0:
            continue

        inputs_bd = make_inputs_bd(netG, inputs, opt)

        denorm_inputs = denormalizer(inputs)
        save_images(cln_save_dir, denorm_inputs, ids)

        denorm_inputs_bd = denormalizer(inputs_bd)
        save_images(bd_save_dir, denorm_inputs_bd, ids)

        total_imgs += len(inputs)
        pbar.set_postfix({"Total images": total_imgs})


def main():
    opt = config.get_arguments().parse_args()
    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 13
    elif opt.dataset == "mnist":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_workers = 40
        opt.num_classes = 8
    elif(opt.dataset == 'imagenet10'):
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
        opt.num_workers = 40
        opt.num_classes = 10
        opt.bs = 32
    else:
        raise Exception("Invalid Dataset")

    opt.pc = 1.0
    opt.bs = 128

    # Dataset
    train_dl = get_dataloader(opt, True, shuffle=False)
    test_dl = get_dataloader(opt, False, shuffle=False)

    # prepare model
    netG = get_model(opt)

    # Load pretrained model
    mode = opt.saving_prefix
    opt.ckpt_folder = os.path.join(opt.checkpoints, "{}_clean".format(mode), opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_clean.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    opt.img_dir = os.path.join(opt.ckpt_folder, "imgs")

    shutil.rmtree(opt.img_dir, ignore_errors=True)
    create_dir(opt.log_dir)
    create_dir(opt.img_dir)

    # Load G
    load_path = os.path.join(opt.checkpoints, opt.load_checkpoint, opt.dataset, "{}_{}.pth.tar".format(opt.dataset, opt.load_checkpoint))
    if not os.path.exists(load_path):
        print("Error: {} not found".format(load_path))
        exit()
    else:
        state_dict = torch.load(load_path)
        netG.load_state_dict(state_dict["netG"])
        netG.eval()

    with torch.no_grad():
        infer(netG, train_dl, opt, "train")
        infer(netG, test_dl, opt, "test")


if __name__ == "__main__":
    main()
