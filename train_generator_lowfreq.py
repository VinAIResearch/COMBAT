import os
import shutil
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as fn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
import timm
from vit_pytorch import SimpleViT
from torchvision.models import efficientnet_b0

import config
from classifier_models import (
    VGG,
    DenseNet121,
    MobileNetV2,
    PreActResNet10,
    PreActResNet18,
    PreActResNetDropout18,
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
    NetC_MNIST,
    NetC_MNIST2,
    NetC_MNIST3,
    Normalizer,
    UnetGeneratorDCT,
)
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from utils.dct import dct_2d, idct_2d


class ViT(SimpleViT):
    # Adapter for SimpleViT
    def __init__(self, input_size=32, n_input=3, *args, **kwargs):
        super().__init__(image_size=input_size, channels=n_input, *args, **kwargs)


def vit_tiny(num_classes=10, n_input=3, input_size=32, **kwargs):
    """ViT-Tiny (Vit-Ti)"""
    patch_size = input_size // 16
    model_kwargs = dict(num_classes=num_classes, img_size=input_size, patch_size=patch_size, in_chans=n_input, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = timm.models.vision_transformer._create_vision_transformer("vit_tiny_patch16_224", pretrained=False, **model_kwargs)
    return model


def vit_small(num_classes=10, n_input=3, input_size=32, **kwargs):
    """ViT-Small (ViT-S)"""
    patch_size = input_size // 16
    model_kwargs = dict(num_classes=num_classes, img_size=input_size, patch_size=patch_size, in_chans=n_input, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = timm.models.vision_transformer._create_vision_transformer("vit_small_patch16_224", pretrained=False, **model_kwargs)
    return model


def vit_base(num_classes=10, n_input=3, input_size=32, **kwargs):
    """ViT-Base (ViT-B)"""
    patch_size = input_size // 16
    model_kwargs = dict(num_classes=num_classes, img_size=input_size, patch_size=patch_size, in_chans=n_input, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = timm.models.vision_transformer._create_vision_transformer("vit_base_patch16_224", pretrained=False, **model_kwargs)
    return model


C_MAPPING_NAMES = {
    "preactresnet10": PreActResNet10,
    "preactresnetdropout18": PreActResNetDropout18,
    "vgg13": partial(VGG, "VGG13"),
    "mobilenetv2": MobileNetV2,
    "efficientnetb0": efficientnet_b0,
    "vit": partial(ViT, patch_size=4, dim=768, depth=6, heads=8, mlp_dim=1024),
    "simplevitsmall8": partial(ViT, patch_size=8, dim=384, depth=12, heads=6, mlp_dim=384 * 4),
    "simplevitsmall4": partial(ViT, patch_size=4, dim=384, depth=12, heads=6, mlp_dim=384 * 4),
    "simplevitsmall2": partial(ViT, patch_size=2, dim=384, depth=12, heads=6, mlp_dim=384 * 4),
    "simplevitbase8": partial(ViT, patch_size=8, dim=768, depth=12, heads=12, mlp_dim=768 * 4),
    "simplevitbase4": partial(ViT, patch_size=4, dim=768, depth=12, heads=12, mlp_dim=768 * 4),
    "simplevitbase2": partial(ViT, patch_size=2, dim=768, depth=12, heads=12, mlp_dim=768 * 4),
    "vittiny": vit_tiny,
    "vitsmall": vit_small,
    "vitbase": vit_base,
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


def create_inputs_bd(inputs, netG, opt, return_noise=False):
    g_out_size = int(opt.input_height * opt.r)

    noise_dct = netG(inputs)  # noise_dct in range [-1, 1]
    noise_bd_dct = torch.zeros_like(inputs)
    noise_bd_dct[:, :, :g_out_size, :g_out_size] = noise_dct

    noise_bd = idct_2d(noise_bd_dct)
    noise_bd = ((noise_bd  - noise_bd.min()) / (noise_bd.max() - noise_bd.min())) * 2 - 1  # Normalize to [-1, 1]
    if opt.dataset == "gtsrb":
        inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate * opt.scale_noise_rate, -1, 1)
    else:
        inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)

    if return_noise:
        return inputs_bd, noise_bd
    else:
        return inputs_bd


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None
    netG = None
    optimizerG = None
    schedulerG = None
    netF = None

    g_in_size = opt.input_height
    g_out_size = int(opt.input_height * opt.r)

    if opt.dataset == "cifar10":
        netC = PreActResNet18().to(opt.device)
        netG = UnetGeneratorDCT(opt, in_size=g_in_size, out_size=g_out_size).to(opt.device)
    if opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = UnetGeneratorDCT(opt, in_size=g_in_size, out_size=g_out_size).to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST3().to(opt.device)
        netG = UnetGeneratorDCT(opt, in_size=g_in_size, out_size=g_out_size).to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = UnetGeneratorDCT(opt, in_size=g_in_size, out_size=g_out_size).to(opt.device)
    if(opt.dataset == 'imagenet10'):
        netC = ResNet18(num_classes=opt.num_classes, input_size=in_size).to(opt.device)
        netG = UnetGeneratorDCT(opt, in_size=g_in_size, out_size=g_out_size).to(opt.device)

    if opt.model != "default":
        netC = C_MAPPING_NAMES[opt.model](num_classes=opt.num_classes, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)

    # Frequency Detector
    F_MAPPING_NAMES["original_dropout"] = partial(FrequencyModelDropout, dropout=opt.F_dropout)
    F_MAPPING_NAMES["original_dropout_ensemble"] = partial(FrequencyModelDropoutEnsemble, dropout=opt.F_dropout, num_ensemble=opt.F_num_ensemble)

    netF = F_MAPPING_NAMES[opt.F_model](num_classes=2, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4, nesterov=True)
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    optimizerG = torch.optim.SGD(netG.parameters(), opt.lr_G, momentum=0.9, weight_decay=5e-4, nesterov=True)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, opt.schedulerG_milestones, opt.schedulerG_lambda)

    return netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, netF


def train(
    netC,
    optimizerC,
    schedulerC,
    netG,
    optimizerG,
    schedulerG,
    netF,
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
    total_loss_l2 = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_F_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()
    criterion_L2 = torch.nn.MSELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt)

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        bd_targets = create_targets_bd(targets, opt)

        ### Train C
        netG.eval()
        netC.train()
        optimizerC.zero_grad()

        # Create backdoor data
        trg_ind = (targets == bd_targets).nonzero()[:, 0]  # Target-label image indices
        ntrg_ind = (targets != bd_targets).nonzero()[:, 0]  # Nontarget-label image indices
        num_bd = np.sum(np.random.rand(trg_ind.shape[0]) < rate_bd)
        inputs_toChange = inputs[trg_ind[:num_bd]]
        if num_bd > 0:
            inputs_bd = create_inputs_bd(inputs_toChange, netG, opt)
        else:
            inputs_bd = inputs[[]]
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

        ### Train G
        netC.eval()
        netG.train()
        optimizerG.zero_grad()

        # Create backdoor data
        inputs_bd, noise_bd = create_inputs_bd(inputs, netG, opt, return_noise=True)
        pred_clean = netC(transforms(inputs))
        pred_bd = netC(transforms(inputs_bd))

        # Classification loss
        loss_ce = criterion_CE(pred_bd, bd_targets)
        loss_l2 = criterion_L2(inputs_bd, inputs)

        # Loss F
        inputs_F = dct_2d(((inputs_bd + 1) / 2 * 255).byte())
        pred_F = netF(inputs_F)

        loss = loss_ce + opt.L2_weight * loss_l2
        loss.backward()
        optimizerG.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()
        total_loss_l2 += loss_l2.detach()
        total_clean_correct += torch.sum(torch.argmax(pred_clean, dim=1) == targets)
        total_bd_correct += torch.sum(torch.argmax(pred_bd, dim=1) == bd_targets)
        total_F_correct += torch.sum(torch.argmax(pred_F, dim=1) == torch.ones_like(targets))

        avg_acc_clean = total_clean_correct * 100.0 / total_sample
        avg_acc_bd = total_bd_correct * 100.0 / total_sample
        avg_acc_F = total_F_correct * 100.0 / total_sample
        avg_loss_ce = total_loss_ce / total_sample
        avg_loss_l2 = total_loss_l2 / total_sample
        progress_bar(
            batch_idx,
            len(train_dl),
            "Clean Acc: {:.4f} | Bd Acc: {:.4f} | F Acc: {:.4f}".format(
                avg_acc_clean,
                avg_acc_bd,
                avg_acc_F,
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
                "L2 Loss": avg_loss_l2,
            },
            epoch,
        )

    if not epoch % 1:
        batch_img = torch.cat([inputs, inputs_bd, noise_bd], dim=2)
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
    test_dl,
    best_clean_acc,
    best_bd_acc,
    best_F_acc,
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
            inputs_bd = create_inputs_bd(inputs_toChange, netG, opt)
            targets_bd = create_targets_bd(targets_toChange, opt)
            preds_bd = netC(inputs_bd)

            total_bd_sample += len(ntrg_ind)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            # Evaluate against Frequency Defense
            inputs_F = dct_2d(((inputs_bd + 1) / 2 * 255).byte())
            targets_F = torch.ones_like(targets_toChange)
            preds_F = netF(inputs_F)
            total_F_correct += torch.sum(torch.argmax(preds_F, 1) == targets_F)

            acc_clean = total_clean_correct * 100.0 / total_clean_sample
            acc_bd = total_bd_correct * 100.0 / total_bd_sample
            acc_F = total_F_correct * 100.0 / total_bd_sample

            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | F Acc: {:.4f} - Best: {:.4f}".format(
                acc_clean,
                best_clean_acc,
                acc_bd,
                best_bd_acc,
                acc_F,
                best_F_acc,
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
            },
            epoch,
        )

    # Save checkpoint
    if acc_clean > best_clean_acc or (acc_clean == best_clean_acc and acc_bd > best_bd_acc):
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        best_F_acc = acc_F
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "netG": netG.state_dict(),
            "schedulerG": schedulerG.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "best_clean_acc": acc_clean,
            "best_bd_acc": acc_bd,
            "best_F_acc": acc_F,
            "epoch_current": epoch,
        }
        torch.save(state_dict, opt.ckpt_path)
    return (
        best_clean_acc,
        best_bd_acc,
        best_F_acc,
    )


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
        if opt.num_classes != 43:
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
        opt.num_classes = 10
        opt.bs = 32
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, netF = get_model(opt)

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

            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            best_F_acc = state_dict["best_F_acc"]
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
            train_dl,
            tf_writer,
            epoch,
            opt,
        )
        (best_clean_acc, best_bd_acc, best_F_acc) = eval(
            netC,
            optimizerC,
            schedulerC,
            netG,
            optimizerG,
            schedulerG,
            netF,
            test_dl,
            best_clean_acc,
            best_bd_acc,
            best_F_acc,
            tf_writer,
            epoch,
            opt,
        )


if __name__ == "__main__":
    main()
