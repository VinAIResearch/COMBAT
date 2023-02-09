import math
import os
import random
import shutil
import sys

import albumentations
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dataloader import get_dataloader
from model import FrequencyModel
from scipy.fftpack import dct, idct
from torch.nn import functional as F

import config

sys.path.insert(0, "../..")
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import efficientnet_b0, googlenet, squeezenet1_0

from classifier_models import VGG, DenseNet121, MobileNetV2, ResNet18
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


def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def idct2(block):
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def valnear0(dct_ori, rmin=-1.5, rmax=1.5):
    return len(dct_ori[dct_ori < rmax][dct_ori[dct_ori < rmax] > rmin])


def addnoise(img):
    aug = albumentations.GaussNoise(p=1, mean=25, var_limit=(10, 70))
    augmented = aug(image=(img * 255).astype(np.uint8))
    auged = augmented["image"] / 255
    return auged


def randshadow(img, input_size=64):
    aug = albumentations.RandomShadow(p=1)
    test = (img * 255).astype(np.uint8)
    augmented = aug(image=cv2.resize(test, (input_size, input_size)))
    auged = augmented["image"] / 255
    return auged


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def tensor2img(t):
    t_np = t.detach().cpu().numpy().transpose(1, 2, 0)
    return t_np


def gauss_smooth(image, sig=6):
    size_denom = 5.0
    sigma = sig * size_denom
    kernel_size = sigma
    mgrid = np.arange(kernel_size, dtype=np.float32)
    mean = (kernel_size - 1.0) / 2.0
    mgrid = mgrid - mean
    mgrid = mgrid * size_denom
    kernel = 1.0 / (sigma * math.sqrt(2.0 * math.pi)) * np.exp(-(((mgrid - 0.0) / (sigma)) ** 2) * 0.5)
    kernel = kernel / np.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernelx = np.tile(np.reshape(kernel, (1, 1, int(kernel_size), 1)), (3, 1, 1, 1))
    kernely = np.tile(np.reshape(kernel, (1, 1, 1, int(kernel_size))), (3, 1, 1, 1))

    padd0 = int(kernel_size // 2)
    evenorodd = int(1 - kernel_size % 2)

    pad = torch.nn.ConstantPad2d((padd0 - evenorodd, padd0, padd0 - evenorodd, padd0), 0.0)
    in_put = torch.from_numpy(np.expand_dims(np.transpose(image.astype(np.float32), (2, 0, 1)), axis=0))
    output = pad(in_put)

    weightx = torch.from_numpy(kernelx)
    weighty = torch.from_numpy(kernely)
    conv = F.conv2d
    output = conv(output, weightx, groups=3)
    output = conv(output, weighty, groups=3)
    output = tensor2img(output[0])

    return output


def patching_train(sample, train_data, n_input=3, input_size=64):
    """
    this code conducts a patching procedure with random white blocks or random noise block
    """
    clean_sample = tensor2img(sample)
    attack = np.random.randint(0, 5)
    pat_size_x = np.random.randint(2, 8)
    pat_size_y = np.random.randint(2, 8)
    output = np.copy(clean_sample)
    if attack == 0:
        block = np.ones((pat_size_x, pat_size_y, n_input))
    elif attack == 1:
        block = np.random.rand(pat_size_x, pat_size_y, n_input)
    elif attack == 2:
        return addnoise(output)
    elif attack == 3:
        return randshadow(output, input_size)
    if attack == 4:
        randind = np.random.randint(train_data.shape[0])
        tri = tensor2img(train_data[randind])
        mid = output + 0.3 * tri
        mid[mid > 1] = 1
        return mid

    margin = np.random.randint(0, 6)
    rand_loc = np.random.randint(0, 4)
    s = input_size
    if rand_loc == 0:
        output[margin : margin + pat_size_x, margin : margin + pat_size_y, :] = block  # upper left
    elif rand_loc == 1:
        output[margin : margin + pat_size_x, s - margin - pat_size_y : s - margin, :] = block
    elif rand_loc == 2:
        output[s - margin - pat_size_x : s - margin, margin : margin + pat_size_y, :] = block
    elif rand_loc == 3:
        output[s - margin - pat_size_x : s - margin, s - margin - pat_size_y : s - margin, :] = block  # right bottom

    output[output > 1] = 1
    return output


def get_model(opt):
    netC = None
    optimizerC = None

    if opt.model in ["original", "original_holdout"]:
        netC = FrequencyModel(num_classes=2, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
        optimizerC = torch.optim.Adadelta(netC.parameters(), lr=0.05, weight_decay=1e-4)
    if opt.model == "vgg13":
        netC = VGG("VGG13", num_classes=2, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
        optimizerC = torch.optim.Adam(netC.parameters(), lr=0.02, weight_decay=1e-4)
    if opt.model == "densenet121":
        netC = DenseNet121(num_classes=2, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
        optimizerC = torch.optim.Adam(netC.parameters(), lr=0.02, weight_decay=1e-4)
    if opt.model == "mobilenetv2":
        netC = MobileNetV2(num_classes=2, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
        optimizerC = torch.optim.Adam(netC.parameters(), lr=0.02, weight_decay=1e-4)
    if opt.model == "resnet18":
        netC = ResNet18(num_classes=2, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
        optimizerC = torch.optim.Adam(netC.parameters(), lr=0.02, weight_decay=1e-4)
    if opt.model == "efficientnetb0":
        netC = efficientnet_b0(num_classes=2).to(opt.device)
        optimizerC = torch.optim.Adam(netC.parameters(), lr=0.02, weight_decay=1e-4)
    if opt.model == "squeezenet":
        netC = squeezenet1_0(num_classes=2).to(opt.device)
        optimizerC = torch.optim.Adam(netC.parameters(), lr=0.02, weight_decay=1e-4)
    if opt.model == "googlenet":
        netC = googlenet(num_classes=2, aux_logits=False).to(opt.device)
        optimizerC = torch.optim.Adam(netC.parameters(), lr=0.02, weight_decay=1e-4)

    return netC, optimizerC


def train(netC, optimizerC, train_dl, tf_writer, epoch, opt):
    torch.autograd.set_detect_anomaly(True)
    print(" Train:")
    netC.train()
    total_loss_ce = 0
    total_correct = 0
    total_sample = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    for batch_idx, (x, y) in enumerate(train_dl):
        optimizerC.zero_grad()
        x, y = x.to(opt.device), y.to(opt.device)
        poi_x = np.zeros((x.shape[0], opt.input_channel, opt.input_height, opt.input_width))
        for i in range(x.shape[0]):
            poi_x[i] = np.transpose(patching_train(x[i], x, opt.input_channel, opt.input_height), (2, 0, 1))
        x_train = x.detach().cpu().numpy()
        x_dct_train = np.vstack((x_train, poi_x))
        y_dct_train = (np.vstack((np.zeros((x_train.shape[0], 1)), np.ones((x_train.shape[0], 1))))).astype(np.uint8)
        for i in range(x_dct_train.shape[0]):
            for channel in range(3):
                x_dct_train[i][channel, :, :] = dct2((x_dct_train[i][channel, :, :] * 255).astype(np.uint8))
        idx = np.arange(x_dct_train.shape[0])
        random.shuffle(idx)
        x_final_train = torch.tensor(x_dct_train[idx], device=opt.device, dtype=torch.float)
        y_final_train = torch.tensor(np.ndarray.flatten(y_dct_train[idx]).astype(int).tolist(), device=opt.device)
        preds = netC(x_final_train)
        loss_ce = criterion_CE(preds, y_final_train)
        if torch.isnan(preds).any() or torch.isnan(y_final_train).any():
            print(preds, y_final_train)
        loss = loss_ce

        loss.backward()
        optimizerC.step()

        total_loss_ce += loss_ce.detach()
        total_correct += torch.sum(torch.argmax(preds, dim=1) == y_final_train)
        total_sample += x_final_train.shape[0]
        avg_acc = total_correct * 100.0 / total_sample
        avg_loss_ce = total_loss_ce / total_sample
        if batch_idx % 100 == 0:
            print(batch_idx, len(train_dl), "CE Loss: {:.4f} | Acc: {:.4f}".format(avg_loss_ce, avg_acc))

    # for tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Accuracy", {"Train": avg_acc}, epoch)
        tf_writer.add_scalar("CE_Loss", avg_loss_ce, epoch)


def eval(netC, optimizerC, test_dl, best_acc, tf_writer, epoch, opt):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_correct = 0

    for batch_idx, (x, y) in enumerate(test_dl):
        with torch.no_grad():
            x, y = x.to(opt.device), y.to(opt.device)
            poi_x = np.zeros((x.shape[0], opt.input_channel, opt.input_height, opt.input_width))
            for i in range(x.shape[0]):
                poi_x[i] = np.transpose(patching_train(x[i], x, opt.input_channel, opt.input_height), (2, 0, 1))
            x_train = x.detach().cpu().numpy()
            x_dct_train = np.vstack((x_train, poi_x))
            y_dct_train = (np.vstack((np.zeros((x_train.shape[0], 1)), np.ones((x_train.shape[0], 1))))).astype(np.uint8)
            for i in range(x_dct_train.shape[0]):
                for channel in range(3):
                    x_dct_train[i][channel, :, :] = dct2((x_dct_train[i][channel, :, :] * 255).astype(np.uint8))
            x_final_train = torch.tensor(x_dct_train, device=opt.device, dtype=torch.float)
            y_final_train = torch.tensor(np.ndarray.flatten(y_dct_train).astype(int).tolist(), device=opt.device)
            preds = netC(x_final_train)
            total_correct += torch.sum(torch.argmax(preds, dim=1) == y_final_train)

            total_sample += x_final_train.shape[0]
            acc = total_correct * 100.0 / total_sample

            info_string = "Acc: {:.4f} - Best: {:.4f}".format(acc, best_acc)
    print(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Accuracy", {"Test": acc}, epoch)

    # Save checkpoint
    if acc > best_acc:
        print(" Saving...")
        best_acc = acc
        state_dict = {"netC": netC.state_dict(), "optimizerC": optimizerC.state_dict(), "best_acc": acc, "epoch_current": epoch}
        torch.save(state_dict, opt.ckpt_path)

    return best_acc


def main():
    opt = config.get_arguments().parse_args()
    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 13
    elif opt.dataset == "mnist":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 1
        opt.num_classes = 10
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_classes = 8
    elif(opt.dataset == 'imagenet10'):
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
        opt.num_classes = 10
    elif(opt.dataset == 'imagenet10small'):
        opt.input_height = 112
        opt.input_width = 112
        opt.input_channel = 3
        opt.num_classes = 10
    elif opt.dataset == "tinyimagenet":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_classes = 200
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    # NOTE: We are using get_dataloader() from `CleanLabelBackdoorGenerator/defenses/frequency_based/dataloader.py`
    # so image tensors are in the range [0, 1]
    print('Loading data...')
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC = get_model(opt)

    # Load pretrained model
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.model)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_detector.pth.tar".format(opt.dataset, opt.model))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    create_dir(opt.log_dir)

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training!!")
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict["netC"])
            optimizerC.load_state_dict(state_dict["optimizerC"])

            best_acc = state_dict["best_acc"]
            epoch_current = state_dict["epoch_current"]

            tf_writer = SummaryWriter(log_dir=opt.log_dir)
            print("Loaded pretrained model")

        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch!!!")
        best_acc = 0.0
        epoch_current = 0
        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        create_dir(opt.log_dir)

        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, train_dl, tf_writer, epoch, opt)
        best_acc = eval(netC, optimizerC, test_dl, best_acc, tf_writer, epoch, opt)


if __name__ == "__main__":
    main()
