import torch
import os
import torch.nn as nn
import copy
from config import get_arguments
import numpy as np
import torch.nn.functional as F

import sys
sys.path.insert(0,'../..')
from utils.dataloader import get_dataloader, PostTensorTransform
from utils.utils import progress_bar
from classifier_models import PreActResNet18
from networks.models import AE, Normalizer, Denormalizer, NetC_MNIST


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
    acc_clean = 0.
    acc_bd = 0.
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0
    
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        total_sample += bs
        
        # Evaluating clean 
        preds_clean = netC(inputs)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100. / total_sample
        
        # Evaluating backdoor
        grid_temps = (identity_grid + opt.scale * noise_grid / opt.input_height) * opt.grid_rescale
        if opt.clamp:
           grid_temps = torch.clamp(grid_temps, -1, 1)
        if opt.nearest > 0:
           grid_temps = (grid_temps + 1)/2 * (inputs.shape[2] - 1) * opt.nearest
           grid_temps = torch.round(grid_temps) / ((inputs.shape[2] - 1) * opt.nearest) * 2 - 1

        inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
        targets_bd = create_targets_bd(targets, opt)
        preds_bd = netC(inputs_bd)
        correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100. / total_sample
        
        progress_bar(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))
    return acc_clean, acc_bd

def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    if(opt.dataset == 'cifar10'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 3 
    elif(opt.dataset == 'gtsrb'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 3
        opt.num_classes = 43
    elif(opt.dataset == 'mnist'):
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel  = 1
    elif(opt.dataset == 'celeba'):
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_workers = 40
    else:
        raise Exception("Invalid Dataset")
    
    # Load models
    if(opt.dataset == 'mnist'):
        netC = NetC_MNIST().to(opt.device)
    else:
        raise Exception("Invalid dataset")
    
    mode = opt.saving_prefix
    path_model = os.path.join(opt.checkpoints, '{}_morph'.format(mode), opt.dataset, '{}_{}_morph.pth.tar'.format(opt.dataset, mode))
    state_dict = torch.load(path_model)
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
    
    # Forward hook for getting layer's output
    container = []
    def forward_hook(module, input, output):
        container.append(output)
    hook = netC.relu4.register_forward_hook(forward_hook)  #relu6
    
    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(test_dl):
        inputs = inputs.to(opt.device)
        netC(inputs)
        progress_bar(batch_idx, len(test_dl))
        
    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()
    
    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    with open(opt.outfile, 'w') as outs:
      for index in range(pruning_mask.shape[0]):
        net_pruned = copy.deepcopy(netC)
        num_pruned = index 
        if(index):
            channel = seq_sort[index]
            pruning_mask[channel] = False
        print("Pruned {} filters".format(num_pruned))
        
        net_pruned.conv4 = nn.Conv2d(64, 64 - num_pruned, (3, 3), 2, 1)  #conv5 (5, 5), 1, 0
        net_pruned.linear6 = nn.Linear(16 * (64 - num_pruned), 512)
        
        # Re-assigning weight to the pruned net
        for name, module in net_pruned._modules.items():
            if('conv4' in name):
                module.weight.data = netC.conv4.weight.data[pruning_mask]
                module.bias.data = netC.conv4.bias.data[pruning_mask]
            elif('linear6' in name):
                module.weight.data = netC.linear6.weight.data.reshape(-1, 64, 16)[:, pruning_mask].reshape(512, -1)
                module.bias.data = netC.linear6.bias.data
            else:
                continue
        clean, bd = eval(net_pruned, identity_grid, noise_grid, test_dl, opt)
        outs.write('%d %0.4f %0.4f\n' % (index, clean, bd))
        
        
if(__name__ == '__main__'):
    main()
