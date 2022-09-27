import config
import torchvision
import torch
import os
import shutil
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as fn

from utils.dataloader import get_dataloader, PostTensorTransform
from utils.utils import progress_bar
from classifier_models import PreActResNet18, PreActResNet10, ResNet18, VGG, MobileNetV2
from networks.models import AE, Normalizer, Denormalizer, NetC_MNIST, NetC_MNIST2, NetC_MNIST3, UnetGenerator
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from torchvision.models import efficientnet_b0
from functools import partial
from vit_pytorch import SimpleViT
import timm


class ViT(SimpleViT):
    # Adapter for SimpleViT
    def __init__(self, input_size=32, patch_size=4, n_input=3, *args, **kwargs):
        patch_size = input_size // 8
        super().__init__(image_size=input_size, patch_size=patch_size, channels=n_input, *args, **kwargs)


def vit_tiny(num_classes=10, n_input=3, input_size=32, **kwargs):
    """ ViT-Tiny (Vit-Ti) """
    patch_size = input_size // 16
    model_kwargs = dict(num_classes=num_classes, img_size=input_size, patch_size=patch_size, in_chans=n_input, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = timm.models.vision_transformer._create_vision_transformer('vit_tiny_patch16_224', pretrained=False, **model_kwargs)
    return model


def vit_small(num_classes=10, n_input=3, input_size=32, **kwargs):
    """ ViT-Small (ViT-S) """
    patch_size = input_size // 16
    model_kwargs = dict(num_classes=num_classes, img_size=input_size, patch_size=patch_size, in_chans=n_input, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = timm.models.vision_transformer._create_vision_transformer('vit_small_patch16_224', pretrained=False, **model_kwargs)
    return model


def vit_base(num_classes=10, n_input=3, input_size=32, **kwargs):
    """ ViT-Base (ViT-B) """
    patch_size = input_size // 16
    model_kwargs = dict(num_classes=num_classes, img_size=input_size, patch_size=patch_size, in_chans=n_input, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = timm.models.vision_transformer._create_vision_transformer('vit_base_patch16_224', pretrained=False, **model_kwargs)
    return model


C_MAPPING_NAMES = {
    "vgg13": partial(VGG, "VGG13"),
    "mobilenetv2": MobileNetV2,
    "efficientnetb0": efficientnet_b0,
    "vit": partial(ViT, dim=768, depth=6, heads=8, mlp_dim=1024),
    "vittiny": vit_tiny,
    "vitsmall": vit_small,
    "vitbase": vit_base,
}


def create_dir(path_dir):
    list_subdir = path_dir.strip('.').split('/')
    list_subdir.remove('')
    base_dir = './'
    for subdir in list_subdir:
        base_dir = os.path.join(base_dir, subdir)
        try:
            os.mkdir(base_dir)
        except:
            pass


def create_targets_bd(targets, opt):
    if(opt.attack_mode == 'all2one'):
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif(opt.attack_mode == 'all2all'):
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def get_model(opt):
    netC = None
    netG = None

    if(opt.dataset == 'cifar10'):
        # Model
        netC = PreActResNet18().to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    if(opt.dataset == 'gtsrb'):
        # Model
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    if(opt.dataset == 'mnist'):
        netC = NetC_MNIST3().to(opt.device) #PreActResNet10(n_input=1).to(opt.device) #NetC_MNIST().to(opt.device)
        netG = UnetGenerator(opt, in_channels=1).to(opt.device)
    if(opt.dataset == 'celeba'):
        netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)

    if opt.model != "default":
        netC = C_MAPPING_NAMES[opt.model](num_classes=opt.num_classes, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)

    return netC, netG


def eval(netC, netG, test_dl, tf_writer, opt):
    print(" Eval:")

    total_clean_sample = 0
    total_bd_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0

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
            if opt.dataset == "gtsrb":
                inputs_bd = torch.clamp(inputs_toChange + noise_bd * opt.noise_rate * opt.scale_noise_rate, -1, 1)
            else:
                inputs_bd = torch.clamp(inputs_toChange + noise_bd * opt.noise_rate, -1, 1)
            targets_bd = create_targets_bd(targets_toChange, opt)
            preds_bd = netC(inputs_bd)

            total_bd_sample += len(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100. / total_clean_sample
            acc_bd = total_bd_correct * 100. / total_bd_sample

            info_string = "Clean Acc: {:.4f} | Bd Acc: {:.4f}".format(acc_clean, acc_bd)
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    tf_writer.add_scalars('Corrected Test Accuracy', {'Clean': acc_clean, 'Bd': acc_bd}, 0)


def main():
    opt = config.get_arguments().parse_args()
    if(opt.dataset == 'cifar10'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 3
    elif(opt.dataset == 'gtsrb'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 3
        opt.num_classes = 13
    elif(opt.dataset == 'mnist'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 1
    elif(opt.dataset == 'celeba'):
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_workers = 40
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, netG = get_model(opt)

    # Load pretrained model
    mode = opt.saving_prefix
    opt.ckpt_folder = os.path.join(opt.checkpoints, '{}_clean'.format(mode), opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, '{}_{}_clean.pth.tar'.format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, 'log_dir')
    create_dir(opt.log_dir)

    # Load G
    load_path = os.path.join(opt.checkpoints, opt.load_checkpoint, opt.dataset, '{}_{}.pth.tar'.format(opt.dataset, opt.load_checkpoint))
    if(not os.path.exists(load_path)):
        print('Error: {} not found'.format(load_path))
        exit()
    else:
        state_dict = torch.load(load_path)
        netC.load_state_dict(state_dict['netC'])
        netC.eval()
        netG.load_state_dict(state_dict['netG'])
        netG.eval()

    tf_writer = SummaryWriter(log_dir=opt.log_dir)

    eval(netC, netG, test_dl, tf_writer, opt)


if(__name__ == '__main__'):
    main()
