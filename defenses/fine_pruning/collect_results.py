import torch
import os
import torch.nn as nn
import copy
from config import get_arguments
import torch.nn.functional as F
import numpy as np

import sys
sys.path.insert(0,'../..')
from utils.dataloader import get_dataloader, PostTensorTransform
from utils.utils import progress_bar
from classifier_models import PreActResNet18, PreActResNet10
from networks.models import AE, Normalizer, Denormalizer, NetC_MNIST, NetC_MNIST3


def create_targets_bd(targets, opt):
    if(opt.attack_mode == 'all2one'):
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif(opt.attack_mode == 'all2all'):
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def eval(netC, identity_grid, noise_grid, test_dl, opt):
    print(" Eval:")
    netC.eval()
    
    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_cross_bd = 0
    total_ae_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs
            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)
            
            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.scale * noise_grid / opt.input_height) * opt.grid_rescale
            if opt.clamp:
                grid_temps = torch.clamp(grid_temps, -1, 1)
            if opt.nearest > 0:
                grid_temps = (grid_temps + 1)/2 * (inputs.shape[2] - 1) * opt.nearest
                grid_temps = torch.round(grid_temps) / ((inputs.shape[2] - 1) * opt.nearest) * 2 - 1

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + 2 * ins / opt.input_height
            if opt.clamp:
                grid_temps2 = torch.clamp(grid_temps2, -1, 1)
            if opt.nearest > 0:
                grid_temps2 = (grid_temps2 + 1)/2 * (inputs.shape[2] - 1) * opt.nearest
                grid_temps2 = torch.round(grid_temps2) / ((inputs.shape[2] - 1) * opt.nearest) * 2 - 1

            inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            targets_bd = create_targets_bd(targets, opt)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            
            inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
            preds_cross = netC(inputs_cross)
            total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)
            total_cross_bd += torch.sum(torch.argmax(preds_cross, 1) == targets_bd)

            acc_clean = total_clean_correct * 100. / total_sample
            acc_bd = total_bd_correct * 100. / total_sample
            acc_cross = total_cross_correct * 100. / total_sample
            bd_cross = total_cross_bd * 100. / total_sample
            
            info_string = "Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross: {:.4f} {:.4f}".format(acc_clean, acc_bd, acc_cross, bd_cross)
            progress_bar(batch_idx, len(test_dl), info_string)


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    if(opt.dataset == 'mnist' or opt.dataset == 'cifar10'):
        opt.num_classes = 10
    elif(opt.dataset == 'gtsrb'):
        opt.num_classes = 43
    else: 
        raise Exception("Invalid Dataset")
    if(opt.dataset == 'cifar10'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 3
    elif(opt.dataset == 'gtsrb'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 3
    elif(opt.dataset == 'mnist'):
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel  = 1
    else:
        raise Exception("Invalid Dataset")
    
    # Load models and masks
    if(opt.dataset == 'cifar10'):
        netC = PreActResNet18().to(opt.device)
    elif(opt.dataset == 'gtsrb'):
        netC = PreActResNet18(num_classes=43).to(opt.device) #NetC_GTRSB().to(opt.device)
    elif(opt.dataset == 'mnist'):
        netC = NetC_MNIST3().to(opt.device)
    else:
        raise Exception("Invalid dataset")

    mode = opt.saving_prefix
    path_model = os.path.join(opt.checkpoints, '{}_morph'.format(mode), opt.dataset, '{}_{}_morph.pth.tar'.format(opt.dataset, mode))
    state_dict = torch.load(path_model)
    print('load C')
    netC.load_state_dict(state_dict['netC'])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    print('load grid')
    identity_grid = state_dict['identity_grid'].to(opt.device)
    noise_grid = state_dict['noise_grid'].to(opt.device)
    print(state_dict['best_clean_acc'], state_dict['best_bd_acc'])

    # Prepare dataloader
    test_dl = get_dataloader(opt, train=False)
    eval(netC, identity_grid, noise_grid, test_dl, opt)

        
if(__name__ == '__main__'):
    main()
