import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as fn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing

import config
from classifier_models import PreActResNet10, PreActResNet18
from networks.models import (AE, CUnetGeneratorv1, Denormalizer, NetC_MNIST,
                             NetC_MNIST2, NetC_MNIST3, Normalizer)
from utils.dataloader_cleanbd import PostTensorTransform, get_dataloader
from utils.utils import progress_bar


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


# def create_bd(inputs, opt):
#    sx = 1.05
#    sy = 1
#    nw = int(inputs.shape[3] * sx)
#    nh = int(inputs.shape[2] * sy)
#    inputs_bd = fn.center_crop(fn.resize(inputs, (nh, nw)), inputs.shape[2:])
#    return inputs_bd


def get_model(opt):
    netC = None
    netG = None

    if opt.dataset == "cifar10":
        # Model
        netC = PreActResNet18().to(opt.device)
        netG = CUnetGeneratorv1(opt).to(opt.device)
    if opt.dataset == "gtsrb":
        # Model
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = CUnetGeneratorv1(opt).to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST3().to(opt.device)  # PreActResNet10(n_input=1).to(opt.device) #NetC_MNIST().to(opt.device)
        netG = CUnetGeneratorv1(opt, in_channels=1).to(opt.device)

    return netC, netG


def eval(netC, netG, test_dl, tf_writer, opt):
    print(" Eval:")
    netC.eval()

    total_clean_sample = 0
    total_bd_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0

    criterion_BCE = torch.nn.BCELoss()
    for batch_idx, (inputs, targets, _) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)

            # Evaluate Clean
            preds_clean = netC(inputs)

            total_clean_sample += len(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            for ci in range(opt.num_classes):
                tmp = targets * 0 + ci
                noise_bd = netG(inputs, tmp)  # + (pattern[None,:,:,:] - inputs) * mask[None, None, :,:]
                inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)
                preds_bd = netC(inputs_bd)

                # Exclude samples with clean label == target label
                ntrg_ind = (targets != tmp).nonzero()[:, 0]
                preds_bd_ntrg = preds_bd[ntrg_ind]
                tmp_ntrg = tmp[ntrg_ind]

                total_bd_sample += len(ntrg_ind)
                total_bd_correct += torch.sum(torch.argmax(preds_bd_ntrg, 1) == tmp_ntrg)

            acc_clean = total_clean_correct * 100.0 / total_clean_sample
            acc_bd = total_bd_correct * 100.0 / total_bd_sample

            info_string = "Clean Acc: {:.4f} | Bd Acc: {:.4f}".format(acc_clean, acc_bd)
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, 0)


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
        opt.num_classes = 43
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_workers = 40
    else:
        raise Exception("Invalid Dataset")

    opt.attack_mode = "all2all"

    # Dataset
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, netG = get_model(opt)

    # Load pretrained model
    mode = opt.saving_prefix
    opt.ckpt_folder = os.path.join(opt.checkpoints, "{}_clean".format(mode), opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_clean.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    create_dir(opt.log_dir)

    load_path = os.path.join(opt.checkpoints, opt.load_checkpoint, opt.dataset, "{}_{}.pth.tar".format(opt.dataset, opt.load_checkpoint))
    if not os.path.exists(load_path):
        print("Error: {} not found".format(load_path))
        exit()
    else:
        state_dict = torch.load(load_path)
        netC.load_state_dict(state_dict["netC"])
        netC.eval()
        netG.load_state_dict(state_dict["netG"])
        netG.eval()

    tf_writer = SummaryWriter(log_dir=opt.log_dir)

    eval(netC, netG, test_dl, tf_writer, opt)


if __name__ == "__main__":
    main()
