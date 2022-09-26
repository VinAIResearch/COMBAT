import torch
import torchvision
import torch.nn.functional as F

from torch.nn import Module
from torch import nn
from .blocks import *
from torchvision import transforms


def smooth_clamp(x, mi, mx):
    tmp = (x-mi)/(mx-mi)
    return mi + (mx-mi)*((1+200**(-tmp+0.5))**(-1) )


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


class Normalizer:
    def __init__(self, opt):
        self.normalizer = self._get_normalizer(opt)

    def _get_normalizer(self, opt):
        if(opt.dataset == 'cifar10'):
            normalizer = Normalize(opt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif(opt.dataset == 'mnist'):
            normalizer = Normalize(opt, [0.5], [0.5])
        elif(opt.dataset == 'gtsrb' or opt.dataset == 'gtsrb2' or opt.dataset == 'celeba'):
            normalizer = Normalize(opt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #None
        else:
            raise Exception("Invalid dataset")
        return normalizer

    def __call__(self, x):
        if(self.normalizer):
            x = self.normalizer(x)
        return x


class Denormalizer:
    def __init__(self, opt):
        self.denormalizer = self._get_denormalizer(opt)

    def _get_denormalizer(self, opt):
        if(opt.dataset == 'cifar10'):
            denormalizer = Denormalize(opt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif(opt.dataset == 'mnist'):
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif(opt.dataset == 'gtsrb' or opt.dataset == 'gtsrb2' or opt.dataset == 'celeba'):
            denormalizer = Denormalize(opt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #None
        else: 
            raise Exception("Invalid dataset")
        return denormalizer

    def __call__(self, x):
        if(self.denormalizer):
            x = self.denormalizer(x)
        return x

#---------------------------- AE ----------------------------#
class Encoder(Module):


    def __init__(self):
        super(Encoder, self).__init__()
        self.downsample1 = Conv2dBlock(3, 12, 4, 2, 1, batch_norm=True, relu=True)
        self.downsample2 = Conv2dBlock(12, 24, 4, 2, 1, batch_norm=True, relu=True)
        self.downsample3 = Conv2dBlock(24, 48, 4, 2, 1, batch_norm=True, relu=True)


    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x
    
    
class Decoder(Module):


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
    

class AE(Module):


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
            denormalizer = Denormalize(opt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        elif(opt.dataset == 'mnist'):
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif(opt.dataset == 'gtsrb' or opt.dataset == 'celeba'):
            denormalizer = Denormalize(opt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else: 
            raise Exception("Invalid dataset")
        return denormalizer


    def _get_normalizer(self, opt):
        if(opt.dataset == 'cifar10'):
            normalizer = Normalize(opt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        elif(opt.dataset == 'mnist'):
            normalizer = Normalize(opt, [0.5], [0.5])
        elif(opt.dataset == 'gtsrb' or opt.dataset == 'gtsrb2' or opt.dataset == 'celeba'):
            normalizer = Normalize(opt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            raise Exception("Invalid dataset")
        return normalizer
    

# class GridGenerator(Module):
#     def __init__(self):
#         super(GridGenerator, self).__init__()
#         self.downsample = Encoder()
#         self.flatten = nn.Flatten()
#         self.linear1 = nn.Linear(48 * 4 * 4, 24 * 4 * 4)
#         self.linear2 = nn.Linear(24 * 4 * 4, 2 * 8 * 8)
#         self.tanh = nn.Tanh()
#
#     def forward(self, x):
#         x = self.downsample(x)
#         x = self.flatten(x)
#         x = self.linear2(self.linear1(x)).view(-1, 2, 8, 8)
#         x = self.tanh(x)
#         x = F.upsample(x, scale_factor=4, mode='bicubic').permute(0, 2, 3, 1)
#         return x
#
#
# class NoiseGenerator(nn.Sequential):
#     def __init__(self, opt, in_channels = 8, steps = 3, channel_init=128):
#         super(NoiseGenerator, self).__init__()
#         self.steps = steps
#         channel_current = in_channels
#         channel_next = channel_init
#         for step in range(steps):
#             self.add_module('upsample_{}'.format(step), nn.Upsample(scale_factor=(2, 2), mode='bilinear'))
#             self.add_module('convblock_up_{}'.format(2 * step), Conv2dBlock(channel_current, channel_current))
#             self.add_module('convblock_up_{}'.format(2 * step + 1), Conv2dBlock(channel_current, channel_next))
#             channel_current = channel_next
#             channel_next = channel_next // 2
#         self.add_module('convblock_up_{}'.format(2 * steps), Conv2dBlock(channel_current, 2, relu=False))
#
#     def forward(self, x):
#         for module in self.children():
#             x = module(x)
#         x = nn.Tanh()(x)
#         return x


class UnetGenerator_bk(Module):
    def __init__(self, opt, in_channels = 3, nf=64, use_bias=True):
        super(UnetGenerator_bk, self).__init__()
        self.act = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        self.conv0_0 = nn.Conv2d(in_channels, nf, kernel_size=3, stride=2, padding=1, bias=use_bias)
        #self.bn0_0 = nn.InstanceNorm2d(nf)
        self.conv0_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn0_1 = nn.InstanceNorm2d(nf)
        self.conv1_0 = nn.Conv2d(nf, nf*2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn1_0 = nn.InstanceNorm2d(nf*2)
        self.conv1_1 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn1_1 = nn.InstanceNorm2d(nf*2)
        self.conv2_0 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn2_0 = nn.InstanceNorm2d(nf*4)
        self.conv2_1 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn2_1 = nn.InstanceNorm2d(nf*4)
        self.conv3_0 = nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn3_0 = nn.InstanceNorm2d(nf*8)
        self.conv3_1 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn3_1 = nn.InstanceNorm2d(nf*8)

        # self.upconv3_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn3_2 = nn.InstanceNorm2d(nf*8)
        self.upconv3_1 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn3_1 = nn.InstanceNorm2d(nf*8)
        self.upconv3_0 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn3_0 = nn.InstanceNorm2d(nf*4)
        # self.upconv2_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn2_2 = nn.InstanceNorm2d(nf*4)
        self.upconv2_1 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn2_1 = nn.InstanceNorm2d(nf*4)
        self.upconv2_0 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn2_0 = nn.InstanceNorm2d(nf*2)
        # self.upconv1_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn1_2 = nn.InstanceNorm2d(nf*2)
        self.upconv1_1 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn1_1 = nn.InstanceNorm2d(nf*2)
        self.upconv1_0 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn1_0 = nn.InstanceNorm2d(nf)
        # self.upconv0_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn0_2 = nn.InstanceNorm2d(nf)
        self.upconv0_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn0_1 = nn.InstanceNorm2d(nf)
        self.upconv0_0 = nn.Conv2d(nf, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.do = nn.Dropout(p=0.3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        f0 = self.conv0_0(x)
        f0 = self.bn0_1(self.conv0_1(self.act(f0)))
        f1 = self.bn1_0(self.conv1_0(self.act(f0)))
        f1 = self.bn1_1(self.conv1_1(self.act(f1)))
        f2 = self.bn2_0(self.conv2_0(self.act(f1)))
        f2 = self.bn2_1(self.conv2_1(self.act(f2)))
        f3 = self.bn3_0(self.conv3_0(self.act(f2)))
        f3 = self.bn3_1(self.conv3_1(self.act(f3)))
        # f3 = self.do(f3)

        # u3 = self.upbn3_2(self.upconv3_2(self.act(self.up(f3))))
        u3 = self.upbn3_1(self.upconv3_1(self.act(self.up(f3))))
        u3 = self.upbn3_0(self.upconv3_0(self.act(u3))) + f2
        # u2 = self.upbn2_2(self.upconv2_2(self.act(self.up(u3))))
        u2 = self.upbn2_1(self.upconv2_1(self.act(self.up(u3))))
        u2 = self.upbn2_0(self.upconv2_0(self.act(u2))) + f1
        # u1 = self.upbn1_2(self.upconv1_2(self.act(self.up(u2))))
        u1 = self.upbn1_1(self.upconv1_1(self.act(self.up(u2))))
        u1 = self.upbn1_0(self.upconv1_0(self.act(u1))) + f0
        # u0 = self.upbn0_2(self.upconv0_2(self.act(self.up(u1))))
        u0 = self.upbn0_1(self.upconv0_1(self.act(self.up(u1))))
        u0 = torch.clamp(self.tanh(self.upconv0_0(self.act(u0))) * 0.08 + x, -1, 1)
        return u0

class UnetGenerator(Module):
    def __init__(self, opt, in_channels = 3, nf=64, use_bias=True, out_channel=None):
        super(UnetGenerator, self).__init__()
        if out_channel is None:
            out_channel = in_channels
        self.act = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        self.conv0_0 = nn.Conv2d(in_channels, nf, kernel_size=3, stride=2, padding=1, bias=use_bias)
        #self.bn0_0 = nn.InstanceNorm2d(nf)
        self.conv0_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn0_1 = nn.InstanceNorm2d(nf)
        self.conv1_0 = nn.Conv2d(nf, nf*2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn1_0 = nn.InstanceNorm2d(nf*2)
        self.conv1_1 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn1_1 = nn.InstanceNorm2d(nf*2)
        self.conv2_0 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn2_0 = nn.InstanceNorm2d(nf*4)
        self.conv2_1 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn2_1 = nn.InstanceNorm2d(nf*4)
        self.conv3_0 = nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn3_0 = nn.InstanceNorm2d(nf*8)
        self.conv3_1 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn3_1 = nn.InstanceNorm2d(nf*8)

        # self.upconv3_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn3_2 = nn.InstanceNorm2d(nf*8)
        self.upconv3_1 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn3_1 = nn.InstanceNorm2d(nf*8)
        self.upconv3_0 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn3_0 = nn.InstanceNorm2d(nf*4)
        # self.upconv2_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn2_2 = nn.InstanceNorm2d(nf*4)
        self.upconv2_1 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn2_1 = nn.InstanceNorm2d(nf*4)
        self.upconv2_0 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn2_0 = nn.InstanceNorm2d(nf*2)
        # self.upconv1_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn1_2 = nn.InstanceNorm2d(nf*2)
        self.upconv1_1 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn1_1 = nn.InstanceNorm2d(nf*2)
        self.upconv1_0 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn1_0 = nn.InstanceNorm2d(nf)
        # self.upconv0_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn0_2 = nn.InstanceNorm2d(nf)
        self.upconv0_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn0_1 = nn.InstanceNorm2d(nf)
        self.upconv0_0 = nn.Conv2d(nf, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.do = nn.Dropout(p=0.3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        f0 = self.conv0_0(x)
        f0 = self.bn0_1(self.conv0_1(self.act(f0)))
        f1 = self.bn1_0(self.conv1_0(self.act(f0)))
        f1 = self.bn1_1(self.conv1_1(self.act(f1)))
        f2 = self.bn2_0(self.conv2_0(self.act(f1)))
        f2 = self.bn2_1(self.conv2_1(self.act(f2)))
        f3 = self.bn3_0(self.conv3_0(self.act(f2)))
        f3 = self.bn3_1(self.conv3_1(self.act(f3)))
        # f3 = self.do(f3)

        # u3 = self.upbn3_2(self.upconv3_2(self.act(self.up(f3))))
        u3 = self.upbn3_1(self.upconv3_1(self.act(self.up(f3))))
        u3 = self.upbn3_0(self.upconv3_0(self.act(u3))) + f2
        # u2 = self.upbn2_2(self.upconv2_2(self.act(self.up(u3))))
        u2 = self.upbn2_1(self.upconv2_1(self.act(self.up(u3))))
        u2 = self.upbn2_0(self.upconv2_0(self.act(u2))) + f1
        # u1 = self.upbn1_2(self.upconv1_2(self.act(self.up(u2))))
        u1 = self.upbn1_1(self.upconv1_1(self.act(self.up(u2))))
        u1 = self.upbn1_0(self.upconv1_0(self.act(u1))) + f0
        # u0 = self.upbn0_2(self.upconv0_2(self.act(self.up(u1))))
        u0 = self.upbn0_1(self.upconv0_1(self.act(self.up(u1))))
        u0 = self.tanh(self.upconv0_0(self.act(u0)))
        return u0

class GridGenerator(Module):
    def __init__(self, opt, in_channels = 3, nf=64, use_bias=True):
        super(GridGenerator, self).__init__()
        self.S = opt.s
        self.act = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        self.conv0_0 = nn.Conv2d(in_channels, nf, kernel_size=3, stride=2, padding=1, bias=use_bias)
        #self.bn0_0 = nn.InstanceNorm2d(nf)
        self.conv0_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn0_1 = nn.InstanceNorm2d(nf)
        self.conv1_0 = nn.Conv2d(nf, nf*2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn1_0 = nn.InstanceNorm2d(nf*2)
        self.conv1_1 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn1_1 = nn.InstanceNorm2d(nf*2)
        self.conv2_0 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn2_0 = nn.InstanceNorm2d(nf*4)
        self.conv2_1 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn2_1 = nn.InstanceNorm2d(nf*4)
        self.conv3_0 = nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn3_0 = nn.InstanceNorm2d(nf*8)
        self.conv3_1 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn3_1 = nn.InstanceNorm2d(nf*8)

        self.fc1 = nn.Linear(nf * 8, nf)
        self.fc2 = nn.Linear(nf, self.S*self.S*2)
        # self.do = nn.Dropout(p=0.3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        f0 = self.conv0_0(x)
        f0 = self.bn0_1(self.conv0_1(self.act(f0)))
        f1 = self.bn1_0(self.conv1_0(self.act(f0)))
        f1 = self.bn1_1(self.conv1_1(self.act(f1)))
        f2 = self.bn2_0(self.conv2_0(self.act(f1)))
        f2 = self.bn2_1(self.conv2_1(self.act(f2)))
        f3 = self.bn3_0(self.conv3_0(self.act(f2)))
        f3 = self.bn3_1(self.conv3_1(self.act(f3)))
        f = F.adaptive_avg_pool2d(f3, 1).squeeze()
        f = self.fc1(f)
        f = self.fc2(self.act(f)).reshape((-1, 2, self.S, self.S))
        f = self.tanh(f)
        return f

class MixedGenerator(Module):
    def __init__(self, opt, in_channels = 3, nf=64, use_bias=True, out_channel=None):
        super(MixedGenerator, self).__init__()
        if out_channel is None:
            out_channel = in_channels
        self.S = opt.s
        self.act = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        self.conv0_0 = nn.Conv2d(in_channels, nf, kernel_size=3, stride=2, padding=1, bias=use_bias)
        #self.bn0_0 = nn.InstanceNorm2d(nf)
        self.conv0_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn0_1 = nn.InstanceNorm2d(nf)
        self.conv1_0 = nn.Conv2d(nf, nf*2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn1_0 = nn.InstanceNorm2d(nf*2)
        self.conv1_1 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn1_1 = nn.InstanceNorm2d(nf*2)
        self.conv2_0 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn2_0 = nn.InstanceNorm2d(nf*4)
        self.conv2_1 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn2_1 = nn.InstanceNorm2d(nf*4)
        self.conv3_0 = nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn3_0 = nn.InstanceNorm2d(nf*8)
        self.conv3_1 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn3_1 = nn.InstanceNorm2d(nf*8)

        self.fc1 = nn.Linear(nf * 8, nf)
        self.fc2 = nn.Linear(nf, self.S*self.S*2)

        # self.upconv3_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn3_2 = nn.InstanceNorm2d(nf*8)
        self.upconv3_1 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn3_1 = nn.InstanceNorm2d(nf*8)
        self.upconv3_0 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn3_0 = nn.InstanceNorm2d(nf*4)
        # self.upconv2_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn2_2 = nn.InstanceNorm2d(nf*4)
        self.upconv2_1 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn2_1 = nn.InstanceNorm2d(nf*4)
        self.upconv2_0 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn2_0 = nn.InstanceNorm2d(nf*2)
        # self.upconv1_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn1_2 = nn.InstanceNorm2d(nf*2)
        self.upconv1_1 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn1_1 = nn.InstanceNorm2d(nf*2)
        self.upconv1_0 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn1_0 = nn.InstanceNorm2d(nf)
        # self.upconv0_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn0_2 = nn.InstanceNorm2d(nf)
        self.upconv0_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn0_1 = nn.InstanceNorm2d(nf)
        self.upconv0_0 = nn.Conv2d(nf, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.tanh = nn.Tanh()

    def forward(self, x):
        f0 = self.conv0_0(x)
        f0 = self.bn0_1(self.conv0_1(self.act(f0)))
        f1 = self.bn1_0(self.conv1_0(self.act(f0)))
        f1 = self.bn1_1(self.conv1_1(self.act(f1)))
        f2 = self.bn2_0(self.conv2_0(self.act(f1)))
        f2 = self.bn2_1(self.conv2_1(self.act(f2)))
        f3 = self.bn3_0(self.conv3_0(self.act(f2)))
        f3 = self.bn3_1(self.conv3_1(self.act(f3)))
        # f3 = self.do(f3)

        # u3 = self.upbn3_2(self.upconv3_2(self.act(self.up(f3))))
        u3 = self.upbn3_1(self.upconv3_1(self.act(self.up(f3))))
        u3 = self.upbn3_0(self.upconv3_0(self.act(u3))) + f2
        # u2 = self.upbn2_2(self.upconv2_2(self.act(self.up(u3))))
        u2 = self.upbn2_1(self.upconv2_1(self.act(self.up(u3))))
        u2 = self.upbn2_0(self.upconv2_0(self.act(u2))) + f1
        # u1 = self.upbn1_2(self.upconv1_2(self.act(self.up(u2))))
        u1 = self.upbn1_1(self.upconv1_1(self.act(self.up(u2))))
        u1 = self.upbn1_0(self.upconv1_0(self.act(u1))) + f0
        # u0 = self.upbn0_2(self.upconv0_2(self.act(self.up(u1))))
        u0 = self.upbn0_1(self.upconv0_1(self.act(self.up(u1))))
        u0 = self.tanh(self.upconv0_0(self.act(u0)))

        f = F.adaptive_avg_pool2d(f3, 1).reshape((f3.shape[0],-1))
        f = self.fc1(f)
        f = self.fc2(self.act(f)).reshape((-1, 2, self.S, self.S))
        f = self.tanh(f)
        return f, u0


class CUnetGeneratorv1(Module):
    def __init__(self, opt, in_channels = 3, nf=64, use_bias=True, out_channel=None):
        super(CUnetGeneratorv1, self).__init__()
        self.num_classes = opt.num_classes
        if out_channel is None:
            out_channel = in_channels
        self.act = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        self.conv0_0 = nn.Conv2d(in_channels, nf, kernel_size=3, stride=2, padding=1, bias=use_bias)
        #self.bn0_0 = nn.InstanceNorm2d(nf)
        self.conv0_1 = nn.Conv2d(nf + self.num_classes, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn0_1 = nn.InstanceNorm2d(nf)
        self.conv1_0 = nn.Conv2d(nf, nf*2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn1_0 = nn.InstanceNorm2d(nf*2)
        self.conv1_1 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn1_1 = nn.InstanceNorm2d(nf*2)
        self.conv2_0 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn2_0 = nn.InstanceNorm2d(nf*4)
        self.conv2_1 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn2_1 = nn.InstanceNorm2d(nf*4)
        self.conv3_0 = nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.bn3_0 = nn.InstanceNorm2d(nf*8)
        self.conv3_1 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn3_1 = nn.InstanceNorm2d(nf*8)

        # self.upconv3_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn3_2 = nn.InstanceNorm2d(nf*8)
        self.upconv3_1 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn3_1 = nn.InstanceNorm2d(nf*8)
        self.upconv3_0 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn3_0 = nn.InstanceNorm2d(nf*4)
        # self.upconv2_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn2_2 = nn.InstanceNorm2d(nf*4)
        self.upconv2_1 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn2_1 = nn.InstanceNorm2d(nf*4)
        self.upconv2_0 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn2_0 = nn.InstanceNorm2d(nf*2)
        # self.upconv1_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn1_2 = nn.InstanceNorm2d(nf*2)
        self.upconv1_1 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn1_1 = nn.InstanceNorm2d(nf*2)
        self.upconv1_0 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn1_0 = nn.InstanceNorm2d(nf)
        # self.upconv0_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn0_2 = nn.InstanceNorm2d(nf)
        self.upconv0_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.upbn0_1 = nn.InstanceNorm2d(nf)
        self.upconv0_0 = nn.Conv2d(nf, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.do = nn.Dropout(p=0.3)
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        f0 = self.conv0_0(x)
        y_emb = F.one_hot(y, num_classes=self.num_classes).float()[:,:,None,None].expand(-1,-1,f0.shape[2],f0.shape[3])
        f0 = torch.cat((f0, y_emb), 1)

        f0 = self.bn0_1(self.conv0_1(self.act(f0)))
        f1 = self.bn1_0(self.conv1_0(self.act(f0)))
        f1 = self.bn1_1(self.conv1_1(self.act(f1)))
        f2 = self.bn2_0(self.conv2_0(self.act(f1)))
        f2 = self.bn2_1(self.conv2_1(self.act(f2)))
        f3 = self.bn3_0(self.conv3_0(self.act(f2)))
        f3 = self.bn3_1(self.conv3_1(self.act(f3)))
        # f3 = self.do(f3)
        #y_emb = F.one_hot(y, num_classes=self.num_classes).float()[:,:,None,None].expand(-1,-1,f3.shape[2],f3.shape[3])
        #f3 = torch.cat((f3, y_emb), 1)

        # u3 = self.upbn3_2(self.upconv3_2(self.act(self.up(f3))))
        u3 = self.upbn3_1(self.upconv3_1(self.act(self.up(f3))))
        u3 = self.upbn3_0(self.upconv3_0(self.act(u3))) + f2
        # u2 = self.upbn2_2(self.upconv2_2(self.act(self.up(u3))))
        u2 = self.upbn2_1(self.upconv2_1(self.act(self.up(u3))))
        u2 = self.upbn2_0(self.upconv2_0(self.act(u2))) + f1
        # u1 = self.upbn1_2(self.upconv1_2(self.act(self.up(u2))))
        u1 = self.upbn1_1(self.upconv1_1(self.act(self.up(u2))))
        u1 = self.upbn1_0(self.upconv1_0(self.act(u1))) + f0
        # u0 = self.upbn0_2(self.upconv0_2(self.act(self.up(u1))))
        u0 = self.upbn0_1(self.upconv0_1(self.act(self.up(u1))))
        u0 = self.tanh(self.upconv0_0(self.act(u0)))
        return u0


### test3
#class UnetGenerator(Module):
#    def __init__(self, opt, in_channels = 3, nf=64, use_bias=True):
#        super(UnetGenerator, self).__init__()
#        self.act = nn.LeakyReLU(0.2, True)
#        self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
#        self.conv0_0 = nn.Conv2d(in_channels, nf, kernel_size=4, stride=2, padding=1, bias=use_bias)
#        #self.bn0_0 = nn.InstanceNorm2d(nf)
#        self.conv0_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.bn0_1 = nn.InstanceNorm2d(nf)
#        self.conv1_0 = nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=use_bias)
#        self.bn1_0 = nn.InstanceNorm2d(nf*2)
#        self.conv1_1 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.bn1_1 = nn.InstanceNorm2d(nf*2)
#        self.conv2_0 = nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=use_bias)
#        self.bn2_0 = nn.InstanceNorm2d(nf*4)
#        self.conv2_1 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.bn2_1 = nn.InstanceNorm2d(nf*4)
#        self.conv3_0 = nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=use_bias)
#        self.bn3_0 = nn.InstanceNorm2d(nf*8)
#        self.conv3_1 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.bn3_1 = nn.InstanceNorm2d(nf*8)

#        self.upconv3_1 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.upbn3_1 = nn.InstanceNorm2d(nf*8)
#        self.upconv3_0 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.upbn3_0 = nn.InstanceNorm2d(nf*4)
#        self.upconv2_1 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.upbn2_1 = nn.InstanceNorm2d(nf*4)
#        self.upconv2_0 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.upbn2_0 = nn.InstanceNorm2d(nf*2)
#        self.upconv1_1 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.upbn1_1 = nn.InstanceNorm2d(nf*2)
#        self.upconv1_0 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.upbn1_0 = nn.InstanceNorm2d(nf)
#        self.upconv0_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.upbn0_1 = nn.InstanceNorm2d(nf)
#        self.upconv0_0 = nn.Conv2d(nf, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)
#        self.tanh = nn.Tanh()

#    def forward(self, x):
#        f0 = self.conv0_0(x)
#        f0 = self.bn0_1(self.conv0_1(self.act(f0)))
#        f1 = self.bn1_0(self.conv1_0(self.act(f0)))
#        f1 = self.bn1_1(self.conv1_1(self.act(f1)))
#        f2 = self.bn2_0(self.conv2_0(self.act(f1)))
#        f2 = self.bn2_1(self.conv2_1(self.act(f2)))
#        f3 = self.bn3_0(self.conv3_0(self.act(f2)))
#        f3 = self.bn3_1(self.conv3_1(self.act(f3)))

#        u3 = self.upbn3_1(self.upconv3_1(self.act(self.up(f3))))
#        u3 = self.upbn3_0(self.upconv3_0(self.act(u3))) + f2
#        u2 = self.upbn2_1(self.upconv2_1(self.act(self.up(u3))))
#        u2 = self.upbn2_0(self.upconv2_0(self.act(u2))) + f1
#        u1 = self.upbn1_1(self.upconv1_1(self.act(self.up(u2))))
#        u1 = self.upbn1_0(self.upconv1_0(self.act(u1))) + f0
#        u0 = self.upbn0_1(self.upconv0_1(self.act(self.up(u1))))
#        u0 = torch.clamp(self.tanh(self.upconv0_0(self.act(u0))) * 0.05 + x, -1, 1)

#        return u0

#---------------------------- Classifiers ----------------------------#

nclasses = 43 # GTSRB as 43 classes
class NetC_GTRSB(nn.Module):
    def __init__(self):
        super(NetC_GTRSB, self).__init__()
        
        self.block1 = Conv2dBlock(3, 32)
        self.block2 = Conv2dBlock(32, 32)
        self.downsample1 = DownSampleBlock(p=0.3)

        self.block3 = Conv2dBlock(32, 64)
        self.block4 = Conv2dBlock(64, 64)
        self.downsample2 = DownSampleBlock(p=0.3)

        self.block5 = Conv2dBlock(64, 128)
        self.block6 = Conv2dBlock(128, 128)
        self.downsample3 = DownSampleBlock(p=0.3)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4 * 4 * 128, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.linear11 = nn.Linear(512, nclasses)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


#class NetC_MNIST(nn.Module):
#    def __init__(self):
#        super(NetC_MNIST, self).__init__()
#        self.conv1 = nn.Conv2d(1, 32, (5, 5), 1, 0)
#        self.relu2 = nn.ReLU(inplace=True)
#        self.dropout3 = nn.Dropout(0.3)

#        self.maxpool4 = nn.MaxPool2d((2, 2))
#        self.conv5 = nn.Conv2d(32, 64, (5, 5), 1, 0)
#        self.relu6 = nn.ReLU(inplace=True)
#        self.dropout7 = nn.Dropout(0.3)

#        self.maxpool5 = nn.MaxPool2d((2, 2))
#        self.flatten = nn.Flatten()
#        self.linear6 = nn.Linear(64 * 4 * 4, 512)
#        self.relu7 = nn.ReLU(inplace=True)
#        self.dropout8 = nn.Dropout(0.3)
#        self.linear9 = nn.Linear(512, 10)

#    def forward(self, x):
#        for module in self.children():
#            x = module(x)
#        return x

#class NetC_MNIST(nn.Module):
#    def __init__(self):
#        super(NetC_MNIST, self).__init__()
#        self.conv1 = nn.Conv2d(1, 32, (5, 5), 1, 0)
#        self.relu2 = nn.ReLU(inplace=True)
#        self.dropout3 = nn.Dropout(0.3)

#        self.maxpool4 = nn.MaxPool2d((2, 2))
#        self.conv5 = nn.Conv2d(32, 64, (5, 5), 1, 0)
#        self.relu6 = nn.ReLU(inplace=True)
#        self.dropout7 = nn.Dropout(0.3)

#        self.maxpool5 = nn.MaxPool2d((2, 2))
#        self.flatten = nn.Flatten()
#        self.linear6 = nn.Linear(64 * 4 * 4, 512)
#        self.relu7 = nn.ReLU(inplace=True)
#        self.dropout8 = nn.Dropout(0.3)
#        self.linear9 = nn.Linear(512, 10)

#    def forward(self, x):
#        for module in self.children():
#            x = module(x)
#        return x

class NetC_MNIST(nn.Module):
    def __init__(self):
        super(NetC_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), 1, 0) #24
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, (3, 3), 2, 1) # 12
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv2d(32, 64, (5, 5), 1, 0) # 8
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(64, 64, (3, 3), 2, 1) # 4
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class MNISTBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(MNISTBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.ind = None

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        #out = self.conv2(F.relu(self.bn2(out)))
        if self.ind is not None:
           out += shortcut[:,self.ind,:,:]
        else:
           out += shortcut
        return out

class NetC_MNIST2(nn.Module):
    def __init__(self):
        super(NetC_MNIST2, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 1) #28
        self.relu1 = nn.ReLU(inplace=True)

        self.layer2 = MNISTBlock(32, 64, 2) # 14
        self.layer3 = MNISTBlock(64, 64, 2) # 7
        self.layer4 = MNISTBlock(64, 64, 2) # 4
        self.dropout4 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class MNISTBlock3(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(MNISTBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.ind = None

        #if stride != 1 or in_planes != planes:
        #    self.shortcut = nn.Sequential(
        #        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        #    )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        #shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        #out = self.conv2(F.relu(self.bn2(out)))
        #if self.ind is not None:
        #   out += shortcut[:,self.ind,:,:]
        #else:
        #   out += shortcut
        return out

class NetC_MNIST3(nn.Module):
    def __init__(self):
        super(NetC_MNIST3, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 2, 1) #14
        self.relu1 = nn.ReLU(inplace=True)

        self.layer2 = MNISTBlock3(32, 64, 2) # 14
        #self.layer3 = MNISTBlock3(64, 64, 2) # 7
        self.layer3 = MNISTBlock3(64, 64, 2) # 4
        #self.dropout4 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


#class NetC_MNIST(nn.Module):
#    def __init__(self):
#        super(NetC_MNIST, self).__init__()
#        self.conv2d_1 = nn.Conv2d(1, 16, (3, 3))
#        self.relu_2 = nn.ReLU(inplace=True)
#        self.conv2d_3 = nn.Conv2d(16, 32, (3, 3))
#        self.relu_4 = nn.ReLU(inplace=True)
#        self.dropout_1 = nn.Dropout(0.3)

#        self.maxpool_5 = nn.MaxPool2d((2, 2))
#        self.backnorm_6 = nn.BatchNorm2d(32)
#        self.conv2d_7 = nn.Conv2d(32, 32, (3, 3))
#        self.relu_8 = nn.ReLU(inplace=True)
#        self.dropout_2 = nn.Dropout(0.3)

#        self.conv2d_9 = nn.Conv2d(32, 64, (3, 3))
#        self.relu_10 = nn.ReLU(inplace=True)
#        self.dropout_3 = nn.Dropout(0.3)
        
#        self.maxpool_11 = nn.MaxPool2d((2, 2))
#        self.batchnorm_12 = nn.BatchNorm2d(64)
#        self.dropout_4 = nn.Dropout(0.3)
#        self.flatten = nn.Flatten()
#        self.linear_12 = nn.Linear(16 * 64, 128)
#        self.dropout_5 = nn.Dropout(0.3)
#        self.linear_13 = nn.Linear(128, 10)

#    def forward(self, x):
#        for module in self.children():
#            x = module(x)
#        return x

class NetC_CelebA(nn.Module):
    def __init__(self):
        super(NetC_CelebA, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 32, (3, 3), 1, 1)
        self.backnorm_2 = nn.BatchNorm2d(32)
        self.relu_3 = nn.ReLU(inplace=True)
        self.dropout_4 = nn.Dropout(0.3)

        self.maxpool_5 = nn.MaxPool2d((2, 2))

        self.conv2d_6 = nn.Conv2d(32, 64, (3, 3), 1, 1)
        self.batchnorm_7 = nn.BatchNorm2d(64)
        self.relu_8 = nn.ReLU(inplace=True)
        self.dropout_9 = nn.Dropout(0.3)
        
        self.maxpool_11 = nn.MaxPool2d((2, 2))

        self.conv2d_13 = nn.Conv2d(64, 64, (3, 3), 1, 1)
        self.backnorm_14 = nn.BatchNorm2d(64)
        self.relu_15 = nn.ReLU(inplace=True)
        self.dropout_16 = nn.Dropout(0.3)

        self.maxpool_17 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()
        self.linear_12 = nn.Linear(64 * 64, 128)
        self.dropout_5 = nn.Dropout(0.3)
        self.linear_13 = nn.Linear(128, 8)
    
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
    

class NetC_CelebA1(nn.Module):
    def __init__(self):
        super(NetC_CelebA1, self).__init__()
        self = torchvision.models.resnet18(pretrained=False)
        self.fc = nn.Linear(512, 8)
    
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
    

def main():
    encoder = GridGenerator()
    a = torch.rand((1, 3, 32, 32))
    a = encoder(a)

    
if(__name__ == '__main__'):
    main()
