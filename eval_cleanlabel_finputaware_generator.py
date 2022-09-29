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
from classifier_models import PreActResNet18, PreActResNet10, ResNet18
from networks.models import AE, Normalizer, Denormalizer, NetC_MNIST, NetC_MNIST2, NetC_MNIST3, UnetGenerator
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing


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


#def create_bd(inputs, opt):
#    sx = 1.05
#    sy = 1
#    nw = int(inputs.shape[3] * sx)
#    nh = int(inputs.shape[2] * sy)
#    inputs_bd = fn.center_crop(fn.resize(inputs, (nh, nw)), inputs.shape[2:])
#    return inputs_bd
        
        
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

    return netC, netG


def eval(netC, netG, test_dl, test_dl2, tf_writer, opt):
    print(" Eval:")
    netC.eval()
    
    total_clean_sample = 0
    total_bd_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    
    l = len(test_dl)
    for batch_idx, batch1, batch2 in zip(range(l), test_dl, test_dl2):
        with torch.no_grad():
            inputs, targets = batch1
            inputs2, targets2 = batch2
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)

            # Evaluate Clean
            preds_clean = netC(inputs)

            total_clean_sample += len(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            ntrg_ind = (targets != opt.target_label).nonzero()[:, 0]
            inputs_toChange = inputs[ntrg_ind]
            targets_toChange = targets[ntrg_ind]
            noise_bd = netG(inputs_toChange)
            inputs_bd = torch.clamp(inputs_toChange + noise_bd * opt.noise_rate, -1, 1)
            targets_bd = create_targets_bd(targets_toChange, opt)
            preds_bd = netC(inputs_bd)

            total_bd_sample += len(ntrg_ind)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            # Evaluate Cross-trigger accuracy
            noise_bd2 = netG(inputs2)
            inputs_bd2 = torch.clamp(inputs + noise_bd2 * opt.noise_rate, -1, 1)
            preds_cross = netC(inputs_bd2)

            # Exclude target-class samples
            preds_cross_ntrg = preds_cross[ntrg_ind]
            targets_ntrg = targets[ntrg_ind]

            total_cross_correct += torch.sum(torch.argmax(preds_cross_ntrg, 1) == targets_ntrg)

            acc_clean = total_clean_correct * 100. / total_clean_sample
            acc_bd = total_bd_correct * 100. / total_bd_sample
            acc_cross = total_cross_correct * 100. / total_bd_sample

            info_string = "Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross Acc: {:.4f}".format(acc_clean, acc_bd, acc_cross)
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    tf_writer.add_scalars('Corrected Test Accuracy', {'Clean': acc_clean, 'Bd': acc_bd, 'Cross': acc_cross}, epoch)


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

    opt.num_workers = 0

    # Dataset 
    test_dl = get_dataloader(opt, False)
    test_dl2 = get_dataloader(opt, False)
        
    # prepare model
    netC, netG = get_model(opt)
        
    # Load pretrained model
    mode = opt.saving_prefix
    opt.ckpt_folder = os.path.join(opt.checkpoints, '{}_clean'.format(mode), opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, '{}_{}_clean.pth.tar'.format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, 'log_dir')
    create_dir(opt.log_dir)

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

    eval(netC, netG, test_dl, test_dl2, tf_writer, opt)
    
    
if(__name__ == '__main__'):
    main()
