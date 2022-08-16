import torch
import os
import torch.nn as nn
import copy
from config import get_arguments
from dataloader import get_dataloader
from utils import progress_bar
from networks.models import NetC_GTRSB, NetC_MNIST, Generator
from classifier_models import PreActResNet18


def get_batch_masks(masks, targets, opt):
    if(opt.attack_mode == 'all2all_dynamic'):
        for index in range(targets.shape[0]):
            masks_added = masks[targets[index]].unsqueeze(0)
            if(not index):
                masks_output = masks_added
            else:
                masks_output = torch.cat((masks_output, masks_added))
    elif(opt.attack_mode == 'all2all' or opt.attack_mode == 'all2one'):
        masks_output = masks
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return masks_output


def create_targets_bd(targets, opt):
    if(opt.attack_mode == 'all2one' or opt.attack_mode == 'all2one_mask'):
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif(opt.attack_mode == 'all2all' or opt.attack_mode == 'all2all_dynamic' or opt.attack_mode == 'all2all_mask'):
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def create_bd(netG, inputs, targets, masks, opt, netM=None):
    bd_targets = create_targets_bd(targets, opt)
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)

    if(opt.attack_mode == 'all2all_mask' or opt.attack_mode == 'all2one_mask'):
        masks_output = netM.correct(netM(inputs)) * opt.mask_transparency
    else:
        masks_output = get_batch_masks(masks, bd_targets, opt)
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets


def eval(netC, netG, test_dl, masks, opt, netM=None):
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
        inputs_bd, targets_bd = create_bd(netG, inputs, targets, masks, opt, netM)
        preds_bd = netC(inputs_bd)
        correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100. / total_sample
        
        progress_bar(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    if(opt.dataset == 'mnist' or opt.dataset == 'cifar10'):
        opt.num_classes = 10
    elif(opt.dataset == 'gtrsb'):
        opt.num_classes = 43
    else: 
        raise Exception("Invalid Dataset")
    if(opt.dataset == 'cifar10'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 3
    elif(opt.dataset == 'gtrsb'):
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
    elif(opt.dataset == 'gtrsb'):
        netC = PreActResNet18(num_classes=43).to(opt.device) #NetC_GTRSB().to(opt.device)
    elif(opt.dataset == 'mnist'):
        netC = NetC_MNIST().to(opt.device)
    else:
        raise Exception("Invalid dataset")
    
    path_model = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, '{}_{}_ckpt.pth'.format(opt.attack_mode, opt.dataset))
    state_dict = torch.load(path_model)
    print('load C')
    netC.load_state_dict(state_dict['netC'])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    print('load G')
    netG = Generator(opt)  
    netG.load_state_dict(state_dict['netG'])
    netG.to(opt.device)
    netG.eval()
    netG.requires_grad_(False)
    print('mask')
    masks = state_dict['masks']
    if (opt.attack_mode == 'all2all_mask' or opt.attack_mode == 'all2one_mask'):
       netM = Generator(opt, out_channels=1)  
       netM.load_state_dict(state_dict['netM'])
       netM.to(opt.device)
       netM.eval()
       netM.requires_grad_(False)
    else:
       netM = None

    # Prepare dataloader
    test_dl = get_dataloader(opt, train=False)
    #test_dl2 = get_dataloader(opt, train=False, smooth=True)    
    
    if True: #for index in range(pruning_mask.shape[0] + 1):
        print('No smoothing')
        eval(netC, netG, test_dl, masks, opt, netM)
        print('Smoothing')
        for s in [2, 4]:
          print('s = ', s)
          test_dl2 = get_dataloader(opt, train=False, smooth=s)
          eval(netC, netG, test_dl2, masks, opt, netM)
        print('Color')
        for c in range(7):
          cc = c + 1
          print('c = ', cc)
          test_dl2 = get_dataloader(opt, train=False, c=cc)
          eval(netC, netG, test_dl2, masks, opt, netM)
        for k in [3, 5, 7]:
          print('k = ', k)
          test_dl2 = get_dataloader(opt, train=False, k=k)
          eval(netC, netG, test_dl2, masks, opt, netM)


        
if(__name__ == '__main__'):
    main()
