import argparse
import cv2
import numpy as np
import torch
import os
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from config import get_arguments

import sys
sys.path.insert(0,'../..')
from utils.dataloader import get_dataloader, PostTensorTransform
from utils.utils import progress_bar
from classifier_models import PreActResNet18, PreActResNet10
from networks.models import AE, Normalizer, Denormalizer, NetC_MNIST, NetC_MNIST3
from torch.autograd import Function
from torchvision import models
from PIL import Image, ImageDraw, ImageFont



class Conv2dBlock(nn.Module):


    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, ker_size, stride, padding)
        if(batch_norm):
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05, affine=True)
        if(relu):
            self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class ConvTranspose2dBlock(nn.Module):


    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(ConvTranspose2dBlock, self).__init__()
        self.convtranpose2d = nn.ConvTranspose2d(in_c, out_c, ker_size, stride, padding)
        if(batch_norm):
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05, affine=True)
        if(relu):
            self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    
class Encoder(nn.Module):


    def __init__(self):
        super(Encoder, self).__init__()
        self.downsample1 = Conv2dBlock(3, 12, 4, 2, 1, batch_norm=True, relu=True)
        self.downsample2 = Conv2dBlock(12, 24, 4, 2, 1, batch_norm=True, relu=True)
        self.downsample3 = Conv2dBlock(24, 48, 4, 2, 1, batch_norm=True, relu=True)


    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x
    
    
class Decoder(nn.Module):


    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample1 = ConvTranspose2dBlock(48, 24, 4, 2, 1, batch_norm=True, relu=True)
        self.upsample2 = ConvTranspose2dBlock(24, 12, 4, 2, 1, batch_norm=True, relu=True)
        self.upsample3 = ConvTranspose2dBlock(12, 3, 4, 2, 1, batch_norm=True, relu=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x
    

class AE(nn.Module):


    def __init__(self, opt):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.normalizer = self._get_normalizer(opt)
        self.denormalizer = self._get_denormalizer(opt)


    def forward(self, x):
        x = self.decoder(self.encoder(x))
        if(self.normalizer):
            x = self.normalizer(x)
        return x


    def _get_denormalizer(self, opt):
        if(opt.dataset == 'cifar10'):
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif(opt.dataset == 'mnist'):
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif(opt.dataset == 'gtrsb' or opt.dataset == 'gtrsb2' or opt.dataset == 'celeba'):
            denormalizer = None
        else: 
            raise Exception("Invalid dataset")
        return denormalizer


    def _get_normalizer(self, opt):
        if(opt.dataset == 'cifar10'):
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif(opt.dataset == 'mnist'):
            normalizer = Normalize(opt, [0.5], [0.5])
        elif(opt.dataset == 'gtrsb' or opt.dataset == 'gtrsb2' or opt.dataset == 'celeba'):
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer
    
    
def text_phantom(text, size):
    # Availability is platform dependent
    font = 'LiberationSans-Regular'

    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size // len(text),
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [size, size], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width) // 2,
              (size - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    return np.asarray(canvas) / 255.0


class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)
    
    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone
    
    
class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)
    
    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone
    
        
def get_normalize(opt):
        if(opt.dataset == 'cifar10'):
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif(opt.dataset == 'mnist'):
            normalizer = Normalize(opt, [0.5], [0.5])
        elif(opt.dataset == 'gtrsb' or opt.dataset == 'gtrsb2'):
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer
    
 
def get_denormalize(opt):
        if(opt.dataset == 'cifar10'):
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif(opt.dataset == 'mnist'):
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif(opt.dataset == 'gtsrb' or opt.dataset == 'gtrsb2'):
            denormalizer = None
        else: 
            raise Exception("Invalid dataset")
        return denormalizer
    
        
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
            print(x.shape)
        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        print(torch.argmax(output, 1))

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def get_model(opt):
        if(opt.dataset == 'mnist'):
            classifier = NetC_MNIST()
        elif(opt.dataset == 'cifar10'):
            classifier = PreActResNet18()
        elif(opt.dataset == 'gtsrb' or opt.dataset == 'gtsrb2'):
            classifier = PreActResNet18(num_classes=43)
        else:
            raise Exception("Invalid Dataset")
        # Load pretrained classifier
        mode = opt.saving_prefix
        print(opt.checkpoints)
        path_model = os.path.join(opt.checkpoints, '{}_morph'.format(mode), opt.dataset, '{}_{}_morph.pth.tar'.format(opt.dataset, mode))
        state_dict = torch.load(path_model)
        classifier.load_state_dict(state_dict['netC'])
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        
        identity_grid = state_dict['identity_grid']
        noise_grid = state_dict['noise_grid']
        #ins = torch.rand(1, 2, opt.S2, opt.S2) * 2 - 1
        #ins = ins / torch.mean(torch.abs(ins))
        #noise_grid = F.upsample(ins, size=opt.input_height, mode='bicubic', align_corners=True).permute(0, 2, 3, 1).to(opt.device)
        #array1d = torch.linspace(-1, 1, steps=opt.input_height)
        #x, y = torch.meshgrid(array1d, array1d)
        #identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)
        
        return classifier.to(opt.device), identity_grid.to(opt.device), noise_grid.to(opt.device)
    
def get_clean_model(opt):
        if(opt.dataset == 'mnist'):
            classifier = NetC_MNIST()
        elif(opt.dataset == 'cifar10'):
            classifier = PreActResNet18()
        elif(opt.dataset == 'gtsrb' or opt.dataset == 'gtsrb2'):
            classifier = PreActResNet18(num_classes=43)
        else:
            raise Exception("Invalid Dataset")
        # Load pretrained classifier
        state_dict = torch.load(opt.clean_path)
        classifier.load_state_dict(state_dict['netC'])
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        return classifier.to(opt.device)
    
def show_cam_on_image(img, mask, idx, opt, prefix = ''):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)/255
    cam = cam / np.max(cam)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(opt.dataset, prefix + "bd{}.png".format(idx)), np.uint8(img))
    cv2.imwrite(os.path.join(opt.dataset, prefix + "cam{}.png".format(idx)), np.uint8(255 * cam))
    cv2.imwrite("heatmap.png", np.uint8(255 * heatmap))
    heatmap = heatmap[:,:,::-1].copy()

    heatmap, img = torch.tensor(heatmap).permute(2, 0, 1), torch.tensor(img / 255.).permute(2, 0, 1)
    # print(heatmap.shape)
    # print(img.shape)
    heatmap, img = F.interpolate(heatmap.unsqueeze(0), scale_factor=4), F.interpolate(img.unsqueeze(0), scale_factor=4)
    return heatmap[0], img[0]
    
    
def padding_inputs(inputs, netG, opt, padding=1):
    outputs = torch.ones((inputs.shape[0], inputs.shape[1], inputs.shape[2] + 2 * padding, inputs.shape[3] + 2 * padding)).to(opt.device)
    outputs = netG.normalize_pattern(outputs)
    outputs[:,:,padding:-padding, padding:-padding] = inputs
    #print(outputs.shape)
    return outputs
    

def create_bd(inputs_clean, identity_gird, noise_grid, opt):
    grid_temps = (identity_gird + opt.scale * noise_grid / opt.input_height) * opt.grid_rescale
    if(opt.clamp):
        grid_temps = torch.clamp(grid_temps, -1, 1)
    if(opt.nearest):
        grid_temps = (grid_temps + 1) / 2 * (inputs_clean.shape[2] - 1) * opt.nearest
        grid_temps = torch.round(grid_temps) / ((inputs_clean.shape[2] - 1) * opt.nearest) * 2 - 1
        
    inputs_bd = F.grid_sample(inputs_clean, grid_temps.repeat(inputs_clean.shape[0], 1, 1, 1), align_corners=True)
    targets_bd = torch.ones(inputs.shape[0]).to(opt.device) * opt.target_label
    
    return inputs_bd, targets_bd


if __name__ == '__main__':
    opt = get_arguments().parse_args()
    if(opt.dataset == 'cifar10'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 3 
    elif(opt.dataset == 'gtsrb' or opt.dataset == 'gtsrb2'):
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
    # args = get_args()
    
    # Load pretrained model
    model, identity_grid, noise_grid = get_model(opt)
    model_clean = get_clean_model(opt)
    denormalizer = get_denormalize(opt)
    
    # Prepare dataset
    if opt.dataset == 'gtsrb':
       opt.dataset = 'gtsrb2'
       dl = get_dataloader(opt, False, True)
       opt.dataset == 'gtsrb'
    else:
       dl = get_dataloader(opt, False, True)
    print(len(dl))
    it = iter(dl)
    inputs, targets = next(it)
    inputs, targets = inputs.to(opt.device), targets.to(opt.device)

    # Create backdoor input
    inputs_bd, _ = create_bd(inputs[:10], identity_grid, noise_grid, opt)
    print(inputs_bd.shape)
    
    grad_cam = GradCam(model=model, feature_module=model.layer3, \
                       target_layer_names=["1"], use_cuda=True)
    grad_cam_clean = GradCam(model=model_clean, feature_module=model_clean.layer3, \
                       target_layer_names=["1"], use_cuda=True)
    bs = inputs_bd.shape[0]
    heatmaps = []
    imgs = []
    cams = []
    
    #for sample_idx in range(bs):
    for idx in range(10):
        input_single = inputs_bd[idx].unsqueeze(0).requires_grad_(True)
        print(input_single.shape)
        if(denormalizer):
            img = denormalizer(input_single).squeeze(0)
        else:
            img = input_single.squeeze(0)
            
        img = img.cpu().detach().numpy() * 255
        print(img.shape)
        img = img.transpose((1, 2, 0)) 

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None
        mask = grad_cam(input_single, target_index)
        heatmap, img = show_cam_on_image(img, mask, idx, opt)
        heatmaps.append(heatmap)
        imgs.append(img)
        

        input_single = inputs[idx].unsqueeze(0).requires_grad_(True)
        print(input_single.shape)
        if(denormalizer):
            img = denormalizer(input_single).squeeze(0)
        else:
            img = input_single.squeeze(0)
            
        img = img.cpu().detach().numpy() * 255
        print(img.shape)
        img = img.transpose((1, 2, 0)) 

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None
        mask = grad_cam_clean(input_single, target_index)
        heatmap, img = show_cam_on_image(img, mask, idx, opt, 'clean_')
    # heatmaps = torch.stack(heatmaps, dim=0)
    # imgs = torch.stack(imgs, dim=0)
    # heatmaps = padding_inputs(heatmaps, pattern_generator, opt)
    # imgs = padding_inputs(imgs, pattern_generator, opt)
    # imgs = torch.cat((imgs, heatmaps), dim=2)

    

        
  
