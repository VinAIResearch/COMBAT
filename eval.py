import os
from functools import partial

import config
import timm
import torch
import torchvision.transforms as T
from classifier_models import VGG, MobileNetV2, PreActResNet18, ResNet18
from networks.models import UnetGenerator
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import get_dataloader
from utils.dct import *
from utils.utils import progress_bar
from vit_pytorch import SimpleViT


class ViT(SimpleViT):
    # Adapter for SimpleViT
    def __init__(self, input_size=32, patch_size=4, n_input=3, *args, **kwargs):
        patch_size = input_size // 8
        super().__init__(image_size=input_size, patch_size=patch_size,
                         channels=n_input, *args, **kwargs)


def vit_small(num_classes=10, n_input=3, input_size=32, **kwargs):
    """ViT-Small (ViT-S)"""
    patch_size = input_size // 16
    model_kwargs = dict(
        num_classes=num_classes,
        img_size=input_size,
        patch_size=patch_size,
        in_chans=n_input,
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs
    )
    model = timm.models.vision_transformer._create_vision_transformer(
        "vit_small_patch16_224", pretrained=False, **model_kwargs
    )
    return model


C_MAPPING_NAMES = {
    "vgg13": partial(VGG, "VGG13"),
    "mobilenetv2": MobileNetV2,
    "vit": partial(ViT, dim=768, depth=6, heads=8, mlp_dim=1024),
    "vitsmall": vit_small,
}


def low_freq(x, opt):
    image_size = opt.input_height
    ratio = opt.ratio
    mask = torch.zeros_like(x)
    mask[:, :, : int(image_size * ratio), : int(image_size * ratio)] = 1
    x_dct = dct_2d((x + 1) / 2 * 255)
    x_dct *= mask
    x_idct = (idct_2d(x_dct) / 255 * 2) - 1
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
        bd_targets = torch.tensor(
            [(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception(
            "{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def get_model(opt):
    netC = None
    netG = None

    if opt.dataset == "cifar10":
        netC = PreActResNet18().to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    elif opt.dataset == "celeba":
        netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    elif opt.dataset == "imagenet10":
        netC = ResNet18(num_classes=opt.num_classes, n_input=opt.input_channel, input_size=opt.input_height).to(
            opt.device
        )
        netG = UnetGenerator(opt).to(opt.device)

    if opt.model != "default":
        netC = C_MAPPING_NAMES[opt.model](
            num_classes=opt.num_classes, n_input=opt.input_channel, input_size=opt.input_height
        ).to(opt.device)

    return netC, netG


def eval(netC, netG, test_dl, tf_writer, opt):
    print(" Eval:")

    total_clean_sample = 0
    total_bd_sample = 0
    total_clean_correct = 0
    total_bd_ba = 0
    total_bd_asr = 0

    gauss_smooth = T.GaussianBlur(kernel_size=opt.kernel_size, sigma=opt.sigma)

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)

            # Evaluate Clean
            preds_clean = netC(inputs)

            total_clean_sample += len(inputs)
            total_clean_correct += torch.sum(
                torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            ntrg_ind = (targets != opt.target_label).nonzero()[:, 0]
            inputs_toChange = inputs[ntrg_ind]
            targets_toChange = targets[ntrg_ind]
            noise_bd = netG(inputs_toChange)
            noise_bd = low_freq(noise_bd, opt)
            inputs_bd = torch.clamp(
                inputs_toChange + noise_bd * opt.noise_rate, -1, 1)
            inputs_bd = gauss_smooth(inputs_bd)
            targets_bd = create_targets_bd(targets_toChange, opt)
            preds_bd = netC(inputs_bd)

            total_bd_sample += len(ntrg_ind)
            total_bd_ba += torch.sum(torch.argmax(preds_bd, 1)
                                     == targets_toChange)
            total_bd_asr += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_clean_sample
            acc_bd_ba = total_bd_ba * 100.0 / total_bd_sample
            acc_bd_asr = total_bd_asr * 100.0 / total_bd_sample

            info_string = "Clean Acc: {:.4f} | Bd BA: {:.4f} | Bd ASR: {:.4f}".format(
                acc_clean, acc_bd_ba, acc_bd_asr)
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    tf_writer.add_scalars("Test Accuracy", {
                          "Clean": acc_clean, "Bd BA": acc_bd_ba, "Bd ASR": acc_bd_asr}, 0)


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
    elif opt.dataset == "imagenet10":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
        opt.num_workers = 40
        opt.num_classes = 10
        opt.bs = 32
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, netG = get_model(opt)

    # Load pretrained model
    mode = opt.saving_prefix
    opt.ckpt_folder = os.path.join(
        opt.checkpoints, "{}_clean".format(mode), opt.dataset)
    opt.ckpt_path = os.path.join(
        opt.ckpt_folder, "{}_{}_clean.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    create_dir(opt.log_dir)

    # Load clean model
    load_path = os.path.join(
        opt.checkpoints,
        opt.load_checkpoint_clean,
        opt.dataset,
        "{}_{}.pth.tar".format(opt.dataset, opt.load_checkpoint_clean),
    )
    if not os.path.exists(load_path):
        print("Error: {} not found".format(load_path))
        exit()
    else:
        state_dict = torch.load(load_path)
        netC.load_state_dict(state_dict["netC"])
        netC.eval()

    # Load G
    load_path = os.path.join(
        opt.checkpoints, opt.load_checkpoint, opt.dataset, "{}_{}.pth.tar".format(
            opt.dataset, opt.load_checkpoint)
    )
    if not os.path.exists(load_path):
        print("Error: {} not found".format(load_path))
        exit()
    else:
        state_dict = torch.load(load_path)
        netG.load_state_dict(state_dict["netG"])
        netG.eval()

    tf_writer = SummaryWriter(log_dir=opt.log_dir)

    eval(netC, netG, test_dl, tf_writer, opt)


if __name__ == "__main__":
    main()
