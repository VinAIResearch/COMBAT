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
from classifier_models import PreActResNet18, PreActResNet10
from networks.models import AE, Normalizer, Denormalizer, UnetGenerator


def create_targets_bd(targets, opt):
    if(opt.attack_mode == 'all2one'):
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif(opt.attack_mode == 'all2all'):
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def eval(netC, netG, test_dl, opt):
    print(" Eval:")
    acc_clean = 0.
    acc_bd = 0.
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0
    
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs
            # Evaluate Clean
            preds_clean = netC(inputs)
            total_correct_clean += torch.sum(torch.argmax(preds_clean, 1) == targets)
            
            # Evaluate Backdoor
            noise_bd = netG(inputs) 
            inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)
            targets_bd = create_targets_bd(targets, opt)
            preds_bd = netC(inputs_bd)
            total_correct_bd += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_correct_clean * 100. / total_sample
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
    if(opt.dataset == 'cifar10'):
        netC = PreActResNet18().to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    elif(opt.dataset == 'gtsrb'):
        netC = PreActResNet18(num_classes=43).to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    else:
        raise Exception("Invalid dataset")
    
    mode = opt.saving_prefix
    path_model = os.path.join(opt.checkpoints, '{}_clean'.format(opt.saving_prefix), opt.dataset, '{}_{}_clean.pth.tar'.format(opt.dataset, opt.saving_prefix))
    state_dict = torch.load(path_model)
    print('load G')
    netG.load_state_dict(state_dict['netG'])
    netG.to(opt.device)
    print('load C')
    netC.load_state_dict(state_dict['netC'])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    print(state_dict['best_clean_acc'], state_dict['best_bd_acc'])

    # Prepare dataloader
    test_dl = get_dataloader(opt, train=False)

    # Forward hook for getting layer's output
    container = []
    def forward_hook(module, input, output):
        container.append(output)
    hook = netC.layer4.register_forward_hook(forward_hook)
    
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
            channel = seq_sort[index - 1]
            pruning_mask[channel] = False
        print("Pruned {} filters".format(num_pruned))
        
        net_pruned.layer4[1].conv2 = nn.Conv2d(pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False)
        net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)
        
        #Re-assigning weight to the pruned net
        for name, module in net_pruned._modules.items():
            if('layer4' in name):
                module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                module[1].ind = pruning_mask
            elif('linear' == name):
                module.weight.data = netC.linear.weight.data[:, pruning_mask]
                module.bias.data = netC.linear.bias.data
            else:
                continue
        net_pruned.to(opt.device)
        clean, bd = eval(net_pruned, netG, test_dl, opt)
        outs.write('%d %0.4f %0.4f\n' % (index, clean, bd))
        
        
if(__name__ == '__main__'):
    main()
