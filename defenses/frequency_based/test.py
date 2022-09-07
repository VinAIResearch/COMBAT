import config
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random
import shutil
import os
import cv2
import albumentations
from scipy.fftpack import dct, idct
import math
import matplotlib.pyplot as plt
from model import FrequencyModel

import sys
sys.path.insert(0,'../..')
from utils.dataloader import get_dataloader
from networks.models import UnetGenerator

def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

def get_model(opt):
    netC = None
    optimizerC = None

    if(opt.model in ['original', 'original_holdout']):
        netC = FrequencyModel(num_classes=opt.num_classes, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
        optimizerC = torch.optim.Adadelta(netC.parameters(), lr=0.05, weight_decay=1e-4)
    if(opt.model == 'vgg13'):
        netC = VGG("VGG13", num_classes=opt.num_classes, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
        optimizerC = torch.optim.Adam(netC.parameters(), lr=0.02, weight_decay=1e-4)
    if(opt.model == 'densenet121'):
        netC = DenseNet121(num_classes=opt.num_classes, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
        optimizerC = torch.optim.Adam(netC.parameters(), lr=0.02, weight_decay=1e-4)
    if(opt.model == 'mobilenetv2'):
        netC = MobileNetV2(num_classes=opt.num_classes, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
        optimizerC = torch.optim.Adam(netC.parameters(), lr=0.02, weight_decay=1e-4)

    return netC, optimizerC

def test(netC, netG, test_dl, opt):
    netC.eval()
    total_correct = 0
    total_sample = 0
    total_poi_sample = 0

    detection = 0
    for batch_idx, (x, y) in enumerate(test_dl):
        with torch.no_grad():
            bs = x.shape[0]
            x, y = x.to(opt.device), y.to(opt.device)
            noise = netG(x)
            poi_x = torch.clamp(x + opt.noise_rate*noise, -1, 1)
            x_test = x.detach().cpu().numpy()
            poi_x_test = poi_x.detach().cpu().numpy()
            x_dct_test = np.vstack((x_test, poi_x_test))
            y_dct_test = (np.vstack((np.zeros((bs,1)),np.ones((bs,1))))).astype(int)
            for i in range(x_dct_test.shape[0]):
                for channel in range(3):
                    x_dct_test[i][channel,:,:] = dct2(((x_dct_test[i][channel,:,:]+np.ones_like(x_dct_test[i][channel,:,:]))/2*255).astype(np.uint8))
            x_final_test = torch.tensor(x_dct_test, device=opt.device, dtype=torch.float)
            y_final_test = torch.tensor(np.ndarray.flatten(y_dct_test).astype(int).tolist(), device=opt.device)
            preds = netC(x_final_test)
            
            detection += torch.sum(torch.argmax(preds[bs:], dim=1) == y_final_test[bs:])


            total_correct += torch.sum(torch.argmax(preds, dim=1) == y_final_test)
            total_poi_sample += bs
            total_sample += x_final_test.shape[0]

    acc = total_correct * 100. / total_sample
    detection_rate = detection * 100. / total_poi_sample
    info_string = "Acc: {:.4f} - Detection rate: {:.4f}".format(acc, detection_rate)
    print(info_string)

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
    elif(opt.dataset == 'mnist'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 1
    elif(opt.dataset == 'celeba'):
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    opt.num_classes = 2

    # Dataset 
    # NOTE: We are using get_dataloader() from `CleanLabelBackdoorGenerator/utils/dataloader.py`
    # so image tensors are in the range [-1, 1]
    test_dl = get_dataloader(opt, False)
        
    # prepare model
    netC, optimizerC = get_model(opt)
        
    # Load pretrained model
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.model)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, '{}_{}_detector.pth.tar'.format(opt.dataset, opt.model))
    opt.log_dir = os.path.join(opt.ckpt_folder, 'log_dir')
    state_dict_C = torch.load(opt.ckpt_path)
    netC.load_state_dict(state_dict_C['netC'])

    # Load G
    netG = UnetGenerator(opt).to(opt.device)
    load_path = os.path.join(opt.load_checkpoint, '{}_clean'.format(opt.saving_prefix), opt.dataset, '{}_{}_clean.pth.tar'.format(opt.dataset, opt.saving_prefix))
    if(not os.path.exists(load_path)):
            print('Error: {} not found'.format(load_path))
            exit()
    else:
            state_dict_G = torch.load(load_path)
            netG.load_state_dict(state_dict_G['netG'])
            netG.eval()
            netG.requires_grad_(False)
            
    test(netC, netG, test_dl, opt) 
    
if(__name__ == '__main__'):
    main()
