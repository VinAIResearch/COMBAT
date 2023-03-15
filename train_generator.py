import os
import shutil
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import torchvision.transforms.functional as fn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
import timm
from vit_pytorch import SimpleViT

import config
from classifier_models import (
    VGG,
    DenseNet121,
    MobileNetV2,
    PreActResNet10,
    PreActResNet18,
    ResNet18,
)
from defenses.frequency_based.model import (
    FrequencyModel,
    FrequencyModelDropout,
    FrequencyModelDropoutEnsemble,
)
from networks.models import (
    AE,
    Denormalizer,
    Normalizer,
    UnetGenerator,
)
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from utils.dct import *


class ViT(SimpleViT):
    # Adapter for SimpleViT
    def __init__(self, input_size=32, n_input=3, *args, **kwargs):
        super().__init__(image_size=input_size, channels=n_input, *args, **kwargs)


C_MAPPING_NAMES = {
    "vgg13": partial(VGG, "VGG13"),
    "mobilenetv2": MobileNetV2,
    "vit": partial(ViT, patch_size=4, dim=768, depth=6, heads=8, mlp_dim=1024),
    "simplevitsmall8": partial(ViT, patch_size=8, dim=384, depth=12, heads=6, mlp_dim=384 * 4),
}


F_MAPPING_NAMES = {
    "original": FrequencyModel,
    "original_holdout": FrequencyModel,
    "original_dropout": FrequencyModelDropout,
    "original_dropout_ensemble": FrequencyModelDropoutEnsemble,
    "vgg13": partial(VGG, "VGG13"),
    "densenet121": DenseNet121,
    "mobilenetv2": MobileNetV2,
    "resnet18": ResNet18,
}


def low_freq(x, opt):
    image_size = opt.input_height
    ratio = opt.ratio
    mask = torch.zeros_like(x)
    mask[:, :, :int(image_size * ratio), :int(image_size * ratio)] = 1
    x_dct = dct_2d((x+1)/2*255)
    x_dct *= mask
    x_idct = (idct_2d(x_dct)/255*2) - 1
    return x_idct


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


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None
    netG = None
    optimizerG = None
    schedulerG = None
    netF = None
    clean_model = None

    if opt.dataset == "cifar10":
        netC = PreActResNet18().to(opt.device)
        clean_model = PreActResNet18().to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    elif opt.dataset == "celeba":
        netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
        clean_model = ResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    elif opt.dataset == 'imagenet10':
        netC = ResNet18(num_classes=opt.num_classes, input_size=opt.input_height).to(opt.device)
        clean_model = ResNet18(num_classes=opt.num_classes, input_size=opt.input_height).to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)

    if opt.model != "default":
        netC = C_MAPPING_NAMES[opt.model](num_classes=opt.num_classes, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
    if opt.model_clean != "default":
        clean_model = C_MAPPING_NAMES[opt.model](num_classes=opt.num_classes, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)

    # Frequency Detector
    F_MAPPING_NAMES["original_dropout"] = partial(FrequencyModelDropout, dropout=opt.F_dropout)
    F_MAPPING_NAMES["original_dropout_ensemble"] = partial(FrequencyModelDropoutEnsemble, dropout=opt.F_dropout, num_ensemble=opt.F_num_ensemble)

    netF = F_MAPPING_NAMES[opt.F_model](num_classes=2, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4, nesterov=True)
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    optimizerG = torch.optim.SGD(netG.parameters(), opt.lr_G, momentum=0.9, weight_decay=5e-4, nesterov=True)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, opt.schedulerG_milestones, opt.schedulerG_lambda)

    return netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, netF, clean_model


def train(
    netC,
    optimizerC,
    schedulerC,
    netG,
    optimizerG,
    schedulerG,
    netF,
    clean_model,
    train_dl,
    tf_writer,
    epoch,
    opt,
):
    torch.autograd.set_detect_anomaly(True)
    print(" Train:")
    netC.train()

    rate_bd = opt.pc
    total_loss_ce = 0
    total_loss_grad_l2 = 0
    total_loss_l2 = 0
    total_clean_model_loss = 0
    total_sample = 0

    total_clean_correct = 0
    total_bd_correct = 0
    total_F_correct = 0
    total_clean_model_correct = 0
    total_clean_model_bd_ba = 0
    total_clean_model_bd_asr = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()
    criterion_L2 = torch.nn.MSELoss()
    gauss_smooth = T.GaussianBlur(kernel_size=opt.kernel_size, sigma=opt.sigma)

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt)

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        bd_targets = create_targets_bd(targets, opt)

        ### Train C
        netG.eval()
        clean_model.eval()
        netC.train()
        optimizerC.zero_grad()
        # Create backdoor data
        trg_ind = (targets == bd_targets).nonzero()[:, 0]  # Target-label image indices
        ntrg_ind = (targets != bd_targets).nonzero()[:, 0]  # Nontarget-label image indices
        num_bd = np.sum(np.random.rand(trg_ind.shape[0]) < rate_bd)
        # num_bd = int(trg_ind.shape[0] * rate_bd)
        # print(epoch, trg_ind.shape[0], num_bd)
        # if num_bd < 1:
        #   continue
        inputs_toChange = inputs[trg_ind[:num_bd]]
        noise_bd = netG(inputs_toChange)
        if inputs_toChange.shape[0] != 0:
            noise_bd = low_freq(noise_bd, opt)
        inputs_bd = torch.clamp(inputs_toChange + noise_bd * opt.noise_rate, -1, 1)
        if inputs_bd.shape[0] != 0:
            inputs_bd = gauss_smooth(inputs_bd)
        total_inputs = torch.cat([inputs_bd, inputs[trg_ind[num_bd:]], inputs[ntrg_ind]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat(
            [
                bd_targets[trg_ind[:num_bd]],
                targets[trg_ind[num_bd:]],
                targets[ntrg_ind],
            ],
            dim=0,
        )
        total_preds = netC(total_inputs)

        loss_ce = criterion_CE(total_preds, total_targets)
        if torch.isnan(total_preds).any() or torch.isnan(total_targets).any():
            print(total_preds, total_targets)
        loss = loss_ce
        loss.backward()
        optimizerC.step()


        clean_preds = clean_model(transforms(inputs))

        ### Train G
        netC.eval()
        clean_model.eval()
        netG.train()
        optimizerG.zero_grad()

        # Create backdoor data
        noise_bd = netG(inputs)
        noise_bd = low_freq(noise_bd, opt)
        inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)
        inputs_bd = gauss_smooth(inputs_bd)
        pred_clean = netC(transforms(inputs))
        pred_bd = netC(transforms(inputs_bd))

        # Classification loss
        loss_ce = criterion_CE(pred_bd, bd_targets)
        if torch.isnan(total_preds).any() or torch.isnan(total_targets).any():
            print(total_preds, total_targets)
        loss_l2 = criterion_L2(inputs_bd, inputs)  # L2 loss
        inputs_ext = F.pad(inputs, (1, 1, 2, 1))
        inputs_bd_ext = F.pad(inputs_bd, (1, 1, 2, 1))
        loss_grad_l2 = criterion_L2(
            inputs_ext[:, :, 1:] - inputs_ext[:, :, :-1],
            inputs_bd_ext[:, :, 1:] - inputs_bd_ext[:, :, :-1],
        ) + criterion_L2(
            inputs_ext[:, :, :, 1:] - inputs_ext[:, :, :, :-1],
            inputs_bd_ext[:, :, :, 1:] - inputs_bd_ext[:, :, :, :-1],
        )  # Gradient loss

        inputs_F = dct_2d(((inputs_bd + 1) / 2 * 255).byte())
        F_targets = torch.ones_like(targets)
        pred_F = netF(inputs_F)

        # Clean Model Loss
        clean_model_preds = clean_model(transforms(inputs_bd))
        clean_model_loss = criterion_CE(clean_model_preds, targets)

        loss = (
            loss_ce + opt.L2_weight * loss_l2 + opt.clean_model_weight * clean_model_loss
        )  # + loss_grad_l2
        loss.backward()
        optimizerG.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()
        total_loss_l2 += loss_l2.detach()
        total_loss_grad_l2 += loss_grad_l2.detach()
        total_clean_model_loss += clean_model_loss.detach()
        total_clean_correct += torch.sum(torch.argmax(pred_clean, dim=1) == targets)
        total_bd_correct += torch.sum(torch.argmax(pred_bd, dim=1) == bd_targets)
        total_F_correct += torch.sum(torch.argmax(pred_F, dim=1) == F_targets)
        total_clean_model_correct += torch.sum(torch.argmax(clean_preds, dim=1) == targets)
        total_clean_model_bd_ba += torch.sum(torch.argmax(clean_model_preds, dim=1) == targets)
        total_clean_model_bd_asr += torch.sum(torch.argmax(clean_model_preds, dim=1) == bd_targets)

        avg_acc_clean = total_clean_correct * 100.0 / total_sample
        avg_acc_bd = total_bd_correct * 100.0 / total_sample
        avg_acc_F = total_F_correct * 100.0 / total_sample
        avg_clean_model_acc = total_clean_model_correct * 100.0 / total_sample
        avg_clean_model_bd_ba = total_clean_model_bd_ba * 100.0 / total_sample
        avg_clean_model_bd_asr = total_clean_model_bd_asr * 100.0 / total_sample
        avg_loss_ce = total_loss_ce / total_sample
        avg_loss_l2 = total_loss_l2 / total_sample
        avg_loss_grad_l2 = total_loss_grad_l2 / total_sample
        avg_clean_model_loss = total_clean_model_loss / total_sample
        progress_bar(
            batch_idx,
            len(train_dl),
            "Clean Acc: {:.4f} | Bd Acc: {:.4f} | F Acc: {:.4f} | Clean Model Acc: {:.4f} | Clean Model Bd BA: {:.4f} | Clean Model Bd ASR: {:.4f}".format(
                avg_acc_clean,
                avg_acc_bd,
                avg_acc_F,
                avg_clean_model_acc,
                avg_clean_model_bd_ba,
                avg_clean_model_bd_asr,
            ),
        )

    # for tensorboard
    if not epoch % 1:
        tf_writer.add_scalars(
            "Clean Accuracy",
            {
                "Clean": avg_acc_clean,
                "Bd": avg_acc_bd,
                "F": avg_acc_F,
                "CleanModel Acc": avg_clean_model_acc,
                "CleanModel Bd BA": avg_clean_model_bd_ba,
                "CleanModel Bd ASR": avg_clean_model_bd_asr,
                "L2 Loss": avg_loss_l2,
                "Grad L2 Loss": avg_loss_grad_l2,
                "CleanModel Loss": avg_clean_model_loss,
            },
            epoch,
        )

    if not epoch % 20:
        batch_img = torch.cat([inputs, inputs_bd], dim=2)
        if denormalizer is not None:
            batch_img = denormalizer(batch_img)
        grid = torchvision.utils.make_grid(batch_img, normalize=True)
        tf_writer.add_image("Images", grid, global_step=epoch)

    schedulerC.step()
    schedulerG.step()


def eval(
    netC,
    optimizerC,
    schedulerC,
    netG,
    optimizerG,
    schedulerG,
    netF,
    clean_model,
    test_dl,
    best_clean_acc,
    best_bd_acc,
    best_F_acc,
    best_clean_model_acc,
    best_clean_model_bd_ba,
    best_clean_model_bd_asr,
    tf_writer,
    epoch,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_clean_sample = 0
    total_bd_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_F_correct = 0
    total_clean_model_correct = 0
    total_clean_model_bd_ba = 0
    total_clean_model_bd_asr = 0

    gauss_smooth = T.GaussianBlur(kernel_size=opt.kernel_size, sigma=opt.sigma)

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)

            # Evaluate Clean
            preds_clean = netC(inputs)

            total_clean_sample += len(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            ntrg_ind = (targets != opt.target_label).nonzero()[:, 0]
            inputs_toChange = inputs[ntrg_ind]
            targets_toChange = targets[ntrg_ind]
            noise_bd = netG(inputs_toChange)
            noise_bd = low_freq(noise_bd, opt)
            
            inputs_bd = torch.clamp(inputs_toChange + noise_bd * opt.noise_rate, -1, 1)
            inputs_bd = gauss_smooth(inputs_bd)
            targets_bd = create_targets_bd(targets_toChange, opt)
            preds_bd = netC(inputs_bd)

            total_bd_sample += len(ntrg_ind)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            # Evaluate against Frequency Defense
            inputs_F = dct_2d(((inputs_bd + 1) / 2 * 255).byte())
            targets_F = torch.ones_like(targets_toChange)
            preds_F = netF(inputs_F)
            total_F_correct += torch.sum(torch.argmax(preds_F, 1) == targets_F)

            # Evaluate against Clean Model
            clean_model_preds_clean = clean_model(inputs)
            total_clean_model_correct += torch.sum(torch.argmax(clean_model_preds_clean, 1) == targets)
            clean_model_preds_bd = clean_model(inputs_bd)
            total_clean_model_bd_ba += torch.sum(torch.argmax(clean_model_preds_bd, 1) == targets_toChange)
            total_clean_model_bd_asr += torch.sum(torch.argmax(clean_model_preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_clean_sample
            acc_bd = total_bd_correct * 100.0 / total_bd_sample
            acc_F = total_F_correct * 100.0 / total_bd_sample

            acc_clean_model = total_clean_model_correct * 100.0 / total_clean_sample
            bd_ba_clean_model = total_clean_model_bd_ba * 100.0 / total_bd_sample
            bd_asr_clean_model = total_clean_model_bd_asr * 100.0 / total_bd_sample

            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | F Acc: {:.4f} - Best: {:.4f} | Clean Model Acc: {:.4f} - Best: {:.4f} | Clean Model Bd BA: {:.4f} - Best: {:.4f} | Clean Model Bd ASR: {:.4f} - Best: {:.4f}".format(
                acc_clean,
                best_clean_acc,
                acc_bd,
                best_bd_acc,
                acc_F,
                best_F_acc,
                acc_clean_model,
                best_clean_model_acc,
                bd_ba_clean_model,
                best_clean_model_bd_ba,
                bd_asr_clean_model,
                best_clean_model_bd_asr,
            )
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars(
            "Test Accuracy",
            {
                "Clean": acc_clean,
                "Bd": acc_bd,
                "F": acc_F,
                "Clean Model Acc": acc_clean_model,
                "Clean Model Bd BA": bd_ba_clean_model,
                "Clean Model Bd ASR": bd_asr_clean_model,
            },
            epoch,
        )

    # Save checkpoint
    if acc_clean > best_clean_acc or (acc_clean == best_clean_acc and acc_bd > best_bd_acc):
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        best_F_acc = acc_F
        best_clean_model_acc = acc_clean_model
        best_clean_model_bd_ba = bd_ba_clean_model
        best_clean_model_bd_asr = bd_asr_clean_model
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "netG": netG.state_dict(),
            "schedulerG": schedulerG.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "clean_model": clean_model.state_dict(),
            "best_clean_acc": acc_clean,
            "best_bd_acc": acc_bd,
            "best_F_acc": acc_F,
            "best_clean_model_acc": best_clean_model_acc,
            "best_clean_model_bd_ba": best_clean_model_bd_ba,
            "best_clean_model_bd_asr": best_clean_model_bd_asr,
            "epoch_current": epoch,
        }
        torch.save(state_dict, opt.ckpt_path)
    return (
        best_clean_acc,
        best_bd_acc,
        best_F_acc,
        best_clean_model_acc,
        best_clean_model_bd_ba,
        best_clean_model_bd_asr,
    )


def main():
    opt = config.get_arguments().parse_args()
    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_workers = 40
        opt.num_classes = 8
    elif opt.dataset == 'imagenet10':
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
        opt.num_classes = 10
        opt.bs = 32
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, netF, clean_model = get_model(opt)

    # Load pretrained model
    mode = opt.saving_prefix
    opt.ckpt_folder = os.path.join(opt.checkpoints, "{}_clean".format(mode), opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_clean.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    create_dir(opt.log_dir)

    # Load pretrained FrequencyModel
    opt.F_ckpt_folder = os.path.join(opt.F_checkpoints, opt.dataset)
    opt.F_ckpt_path = os.path.join(opt.F_ckpt_folder, opt.F_model, "{}_{}_detector.pth.tar".format(opt.dataset, opt.F_model))
    print(f"Loading {opt.F_model} at {opt.F_ckpt_path}")
    state_dict_F = torch.load(opt.F_ckpt_path)
    netF.load_state_dict(state_dict_F["netC"])
    netF.eval()
    print("Done")


    # Load clean_model
    load_path = os.path.join(opt.checkpoints, opt.load_checkpoint_clean, opt.dataset, "{}_{}.pth.tar".format(opt.dataset, opt.load_checkpoint_clean))
    if not os.path.exists(load_path):
        print("Error: {} not found".format(load_path))
        exit()
    else:
        state_dict = torch.load(load_path)
        clean_model.load_state_dict(state_dict["netC"])
        clean_model.eval()

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training!!")
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict["netC"])
            optimizerC.load_state_dict(state_dict["optimizerC"])
            schedulerC.load_state_dict(state_dict["schedulerC"])
            netG.load_state_dict(state_dict["netG"])
            optimizerG.load_state_dict(state_dict["optimizerG"])
            schedulerG.load_state_dict(state_dict["schedulerG"])
            clean_model.load_state_dict(state_dict["clean_model"])

            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            best_F_acc = state_dict["best_F_acc"]
            best_clean_model_acc = state_dict["best_clean_model_acc"]
            best_clean_model_bd_ba = state_dict["best_clean_model_bd_ba"]
            best_clean_model_bd_asr= state_dict["best_clean_model_bd_asr"]
            epoch_current = state_dict["epoch_current"]

            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_F_acc = 0.0
        best_clean_model_acc = 0.0
        best_clean_model_bd_ba = 0.0
        best_clean_model_bd_asr = 0.0
        epoch_current = 0
        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        create_dir(opt.log_dir)

        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(
            netC,
            optimizerC,
            schedulerC,
            netG,
            optimizerG,
            schedulerG,
            netF,
            clean_model,
            train_dl,
            tf_writer,
            epoch,
            opt,
        )
        (best_clean_acc, best_bd_acc, best_F_acc, best_clean_model_acc, best_clean_model_bd_ba, best_clean_model_bd_asr) = eval(
            netC,
            optimizerC,
            schedulerC,
            netG,
            optimizerG,
            schedulerG,
            netF,
            clean_model,
            test_dl,
            best_clean_acc,
            best_bd_acc,
            best_F_acc,
            best_clean_model_acc,
            best_clean_model_bd_ba,
            best_clean_model_bd_asr,
            tf_writer,
            epoch,
            opt,
        )

if __name__ == "__main__":
    main()
