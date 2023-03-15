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

import config
from classifier_models import PreActResNet10, PreActResNet18
from networks.models import (AE, CUnetGeneratorv1, Denormalizer, NetC_MNIST,
                             NetC_MNIST2, NetC_MNIST3, Normalizer)
from utils.dataloader_cleanbd import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from utils.dct import dct_2d, idct_2d


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


gauss_smooth = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1))


def low_freq(x, opt):
    image_size = opt.input_height
    ratio = opt.ratio
    mask = torch.zeros_like(x)
    mask[:, :, :int(image_size * ratio), :int(image_size * ratio)] = 1
    x_dct = dct_2d((x+1)/2*255)
    x_dct *= mask
    x_idct = (idct_2d(x_dct)/255*2) - 1
    return x_idct


def create_inputs_bd(inputs, targets, netG, opt):
    # Create backdoor data
    noise_bd = netG(inputs, targets)
    if inputs.shape[0] != 0:
        noise_bd = low_freq(noise_bd, opt)
    inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)
    if inputs_bd.shape[0] != 0:
        inputs_bd = gauss_smooth(inputs_bd)
    return inputs_bd


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None
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

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4, nesterov=True)
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    return netC, optimizerC, schedulerC, netG


def train(netC, optimizerC, schedulerC, netG, train_dl, tf_writer, epoch, opt):
    torch.autograd.set_detect_anomaly(True)
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_loss_l2 = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_clean_correct = 0
    total_bd_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()
    criterion_L2 = torch.nn.MSELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt)

    for batch_idx, (inputs, targets, poisoned) in enumerate(train_dl):
        inputs, targets, poisoned = inputs.to(opt.device), targets.to(opt.device), poisoned.to(opt.device)
        bs = inputs.shape[0]

        ### Train f
        # netG.eval()
        netC.train()
        optimizerC.zero_grad()
        # Create backdoor data
        trg_ind = poisoned.nonzero()[:, 0]
        ntrg_ind = (poisoned == False).nonzero()[:, 0]
        num_bd = trg_ind.shape[0]
        # if num_bd < 1:
        #    continue
        inputs_toChange = inputs[trg_ind]
        targets_toChange = targets[trg_ind]
        # noise_bd = netG(inputs_toChange, targets[trg_ind])
        # inputs_bd = torch.clamp(inputs_toChange + noise_bd * opt.noise_rate, -1, 1)
        inputs_bd = create_inputs_bd(inputs_toChange, targets_toChange, netG, opt)
        total_inputs = torch.cat([inputs_bd, inputs[ntrg_ind]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([targets[trg_ind], targets[ntrg_ind]], dim=0)
        total_preds = netC(total_inputs)

        loss_ce = criterion_CE(total_preds, total_targets)
        if torch.isnan(total_preds).any() or torch.isnan(total_targets).any():
            print(total_preds, total_targets)
        loss = loss_ce
        loss.backward()
        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()
        total_clean_correct += torch.sum(torch.argmax(total_preds, dim=1) == total_targets)

        avg_acc_clean = total_clean_correct * 100.0 / total_sample
        avg_loss_ce = total_loss_ce / total_sample
        progress_bar(batch_idx, len(train_dl), "CE Loss: {:.4f} | Clean Acc: {:.4f}".format(avg_loss_ce, avg_acc_clean))

        # Save image for debugging
        if not batch_idx % 5 and num_bd >= 1:
            if not os.path.exists(opt.temps):
                create_dir(opt.temps)
            # path = os.path.join(opt.temps, 'backdoor_image.png')
            batch_img = torch.cat([inputs_toChange, inputs_bd], dim=2)
            if denormalizer is not None:
                batch_img = denormalizer(batch_img)
            grid = torchvision.utils.make_grid(batch_img, normalize=True)

    # for tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Clean Accuracy", {"Clean": avg_acc_clean}, epoch)
        tf_writer.add_image("Images", grid, global_step=epoch)

    schedulerC.step()


def eval(netC, optimizerC, schedulerC, netG, test_dl, best_clean_acc, best_bd_acc, tf_writer, epoch, opt):
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
                # noise_bd = netG(inputs, tmp)  # + (pattern[None,:,:,:] - inputs) * mask[None, None, :,:]
                # inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)
                inputs_bd = create_inputs_bd(inputs, tmp, netG, opt)
                preds_bd = netC(inputs_bd)

                # Exclude samples with clean label == target label
                ntrg_ind = (targets != tmp).nonzero()[:, 0]
                preds_bd_ntrg = preds_bd[ntrg_ind]
                tmp_ntrg = tmp[ntrg_ind]

                total_bd_sample += len(ntrg_ind)
                total_bd_correct += torch.sum(torch.argmax(preds_bd_ntrg, 1) == tmp_ntrg)

            acc_clean = total_clean_correct * 100.0 / total_clean_sample
            acc_bd = total_bd_correct * 100.0 / total_bd_sample

            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(acc_clean, best_clean_acc, acc_bd, best_bd_acc)
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

    # Save checkpoint
    if acc_clean > best_clean_acc:
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        state_dict = {"netC": netC.state_dict(), "schedulerC": schedulerC.state_dict(), "optimizerC": optimizerC.state_dict(), "netG": netG.state_dict(), "best_clean_acc": acc_clean, "best_bd_acc": acc_bd, "epoch_current": epoch}
        torch.save(state_dict, opt.ckpt_path)
    return best_clean_acc, best_bd_acc


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
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC, netG = get_model(opt)

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
        netG.load_state_dict(state_dict["netG"])
        netG.eval()
        tf_writer = SummaryWriter(log_dir=opt.log_dir)
    best_clean_acc = 0.0
    best_bd_acc = 0.0
    epoch_current = 0
    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, netG, train_dl, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc = eval(netC, optimizerC, schedulerC, netG, test_dl, best_clean_acc, best_bd_acc, tf_writer, epoch, opt)


if __name__ == "__main__":
    main()
