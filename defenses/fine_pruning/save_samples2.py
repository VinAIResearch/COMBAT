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

        grid_temps = (identity_grid + opt.scale * noise_grid / opt.input_height) * opt.grid_rescale
        if opt.clamp:
                grid_temps = torch.clamp(grid_temps, -1, 1)
        if opt.nearest > 0:
                grid_temps = (grid_temps + 1)/2 * (inputs.shape[2] - 1) * opt.nearest
                grid_temps = torch.round(grid_temps) / ((inputs.shape[2] - 1) * opt.nearest) * 2 - 1

        ins = F.upsample(torch.rand(inputs.shape[0], 2, 4, 4).to(opt.device) * 2 - 1, size=opt.input_height, mode='bicubic', align_corners=True).permute(0, 2, 3, 1).to(opt.device) #torch.rand(inputs.shape[0], opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
        grid_temps2 = grid_temps.repeat(inputs.shape[0], 1, 1, 1) + 2 * ins / opt.input_height
        if opt.clamp:
           grid_temps2 = torch.clamp(grid_temps2, -1, 1)
        if opt.nearest > 0:
           grid_temps2 = (grid_temps2 + 1)/2 * (inputs.shape[2] - 1) * opt.nearest
           grid_temps2 = torch.round(grid_temps2) / ((inputs.shape[2] - 1) * opt.nearest) * 2 - 1

        inputs_bd = F.grid_sample(inputs, grid_temps.repeat(inputs.shape[0], 1, 1, 1), align_corners=True)
        inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
        targets_bd = create_targets_bd(targets, opt)
        step = int(inputs_bd.shape[0]/n_imgs)
        subs1, subs2, subs3 = [], [], []
        for i in range(n_imgs):
           subs1.append(inputs[(i*step):(i*step+1),:,:,:])
           subs2.append(inputs_bd[(i*step):(i*step+1),:,:,:])
           subs3.append(inputs_cross[(i*step):(i*step+1),:,:,:])
        images1, images2, images3 = torch.cat(subs1, dim=0), torch.cat(subs2, dim=0), torch.cat(subs3, dim=0)
        if denormalizer is not None:
           images1 = denormalizer(images1)
           images2 = denormalizer(images2)
           images3 = denormalizer(images3)
        if not os.path.exists(opt.outfile + '/clean'):
           os.makedirs(opt.outfile + '/clean')
        if not os.path.exists(opt.outfile + '/bd'):
           os.makedirs(opt.outfile + '/bd')
        if not os.path.exists(opt.outfile + '/cross'):
           os.makedirs(opt.outfile + '/cross')
        save_image(images1, opt.outfile + '/clean', batch_idx*n_imgs)
        save_image(images2, opt.outfile + '/bd', batch_idx*n_imgs)
        save_image(images3, opt.outfile + '/cross', batch_idx*n_imgs)
        noise0 = noise_grid.permute(0, 3, 1, 2) / 2 + 0.5
        noise0 = torch.cat((noise0, noise0[:,:1,:,:] * 0 + 1), 1)
        save_image(noise0, opt.outfile, 0)
        noise1 = grid_temps.permute(0, 3, 1, 2) / 2 + 0.5
        noise1 = torch.cat((noise1, noise1[:,:1,:,:] * 0 + 1), 1)
        save_image(noise1, opt.outfile, 1)
        noise2 = grid_temps2[::5].permute(0, 3, 1, 2) / 2 + 0.5
        noise2 = torch.cat((noise2, noise2[:,:1,:,:] * 0 + 1), 1)
        save_image(noise2, opt.outfile, 2)
        noise3 = (ins[::5] + noise_grid[0, None, :,:,:]).permute(0, 3, 1, 2) / 2 + 0.5
        noise3 = torch.cat((noise3, noise3[:,:1,:,:] * 0 + 1), 1)
        save_image(noise3, opt.outfile, 100)


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
