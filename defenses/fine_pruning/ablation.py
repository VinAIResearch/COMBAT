import torch
import os
import torch.nn as nn
import copy
from config import get_arguments
import torchvision
import numpy as np
import torch.nn.functional as F
from PIL import Image
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


def save_image(images, root_path, start_count):
    grid = images #F.upsample(images, scale_factor=2)
    for idx, img in enumerate(grid):
        img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(img)
        im.save('%s/%03d.png' % (root_path, start_count + idx))

def eval(netC, test_dl, identity_grid, noise_grid, opt):
    print(" Eval:")
    acc_clean = 0.
    acc_bd = 0.
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0
    n_imgs = 4
    denormalizer = Denormalizer(opt)
    
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        grid_temps0 = (identity_grid + opt.scale * noise_grid / opt.input_height) * opt.grid_rescale
        if opt.clamp:
                grid_temps0 = torch.clamp(grid_temps0, -1, 1)
        inputs_bds = []
        for nearest in [0, 1, 2, 3]:
            if nearest > 0:
                grid_temps = (grid_temps0 + 1)/2 * (inputs.shape[2] - 1) * nearest
                grid_temps = torch.round(grid_temps) / ((inputs.shape[2] - 1) * nearest) * 2 - 1
            else:
                grid_temps = grid_temps0 + 0

            inputs_bds.append(F.grid_sample(inputs, grid_temps.repeat(inputs.shape[0], 1, 1, 1), align_corners=True))
        step = int(inputs.shape[0]/n_imgs)
        inputs = inputs[::step,:,:,:]
        inputs_bds = [x[::step,:,:,:] for x in inputs_bds]
        if denormalizer is not None:
           inputs = denormalizer(inputs)
           inputs_bds = [denormalizer(x) for x in inputs_bds]
        if not os.path.exists(opt.outfile + '/clean'):
           os.makedirs(opt.outfile + '/clean')
        save_image(inputs, opt.outfile + '/clean/', batch_idx * n_imgs)
        for nearest in [0, 1, 2, 3]:
           if not os.path.exists(opt.outfile + '/bd' + str(nearest)):
              os.makedirs(opt.outfile + '/bd' + str(nearest))
           save_image(inputs_bds[nearest], opt.outfile + '/bd' + str(nearest), batch_idx * n_imgs)


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    if(opt.dataset == 'cifar10'):
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel  = 3 
    elif(opt.dataset == 'gtsrb'):
        opt.input_height = 64
        opt.input_width = 64
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
    
    # Load models and masks
    if(opt.dataset == 'cifar10'):
        netC = PreActResNet18().to(opt.device)
    elif(opt.dataset == 'gtsrb'):
        netC = PreActResNet18(num_classes=43).to(opt.device) #NetC_GTRSB().to(opt.device)
    elif(opt.dataset == 'mnist'):
        netC = NetC_MNIST().to(opt.device)
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
    #identity_grid = state_dict['identity_grid'].to(opt.device)
    #noise_grid = state_dict['noise_grid'].to(opt.device)
    ins = torch.rand(1, 2, 4, 4) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.upsample(ins, size=opt.input_height, mode='bicubic', align_corners=True).permute(0, 2, 3, 1).to(opt.device)
    array1d = torch.linspace(-1, 1, steps=opt.input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)
    print(state_dict['best_clean_acc'], state_dict['best_bd_acc'])


    # Prepare dataloader
    test_dl = get_dataloader(opt, train=False)
    eval(netC, test_dl, identity_grid, noise_grid, opt)


        
if(__name__ == '__main__'):
    main()
