from utils.utils import progress_bar
from utils.dct import *
from utils.dataloader import get_dataloader, get_dataset
from networks.models import Denormalizer, UnetGenerator
from classifier_models import PreActResNet18, ResNet18
import os
import sys

import config
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from dataloader import get_dataloader, get_dataset
from torchvision import transforms


sys.path.insert(0, "../..")


class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = (
                x[:, :, channel] - self.expected_values[channel]) / self.variance[channel]
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
            x_clone[:, :, channel] = x[:, :, channel] * \
                self.variance[channel] + self.expected_values[channel]
        return x_clone


def low_freq(x, opt):
    image_size = opt.input_height
    ratio = opt.ratio
    mask = torch.zeros_like(x)
    mask[:, :, : int(image_size * ratio), : int(image_size * ratio)] = 1
    x_dct = dct_2d((x + 1) / 2 * 255)
    x_dct *= mask
    x_idct = (idct_2d(x_dct) / 255 * 2) - 1
    return x_idct


class STRIP:
    def _superimpose(self, background, overlay):
        output = cv2.addWeighted(background, 1, overlay, 1, 0)
        if len(output.shape) == 2:
            output = np.expand_dims(output, 2)
        return output

    def _get_entropy(self, background, dataset, classifier):
        entropy_sum = [0] * self.n_sample
        x1_add = [0] * self.n_sample
        index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
        for index in range(self.n_sample):
            add_image = self._superimpose(
                background, dataset[index_overlay[index]][0])
            add_image = self.normalize(add_image)
            x1_add[index] = add_image

        py1_add = classifier(torch.stack(x1_add).to(self.device))
        py1_add = torch.sigmoid(py1_add).cpu().numpy()
        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
        return entropy_sum / self.n_sample

    def _get_denormalize(self, opt):
        if opt.dataset == "cifar10" or opt.dataset == "celeba" or opt.dataset == "imagenet10":
            denormalizer = Denormalize(opt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if opt.dataset == "cifar10" or opt.dataset == "celeba" or opt.dataset == "imagenet10":
            normalizer = Normalize(opt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            raise Exception("Invalid dataset")
        if normalizer:
            transform = transforms.Compose([transforms.ToTensor(), normalizer])
        else:
            transform = transforms.ToTensor()
        return transform

    def __init__(self, opt):
        super().__init__()
        self.n_sample = opt.n_sample
        self.normalizer = self._get_normalize(opt)
        self.denormalizer = self._get_denormalize(opt)
        self.device = opt.device

    def normalize(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x

    def denormalize(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x

    def __call__(self, background, dataset, classifier):
        return self._get_entropy(background, dataset, classifier)


def strip(opt, mode="clean"):

    # Prepare pretrained classifier
    if opt.dataset == "cifar10":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    elif opt.dataset == "celeba" or opt.dataset == "imagenet10":
        netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
    else:
        raise Exception("Invalid dataset")

    if mode != "clean":
        netG = UnetGenerator(opt).to(opt.device)
        for param in netG.parameters():
            param.requires_grad = False
        netG.eval()

    # Load pretrained model
    ckpt_folder = os.path.join(
        opt.checkpoints, "{}_clean".format(opt.saving_prefix), opt.dataset)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_clean.pth.tar".format(
        opt.dataset, opt.saving_prefix))
    os.path.join(ckpt_folder, "log_dir")

    state_dict = torch.load(ckpt_path)
    netC.load_state_dict(state_dict["netC"])
    if mode != "clean":
        netG.load_state_dict(state_dict["netG"])
    netC.requires_grad_(False)
    netC.eval()
    netC.to(opt.device)

    # Prepare test set
    testset = get_dataset(opt, train=False)
    opt.bs = opt.n_test
    test_dataloader = get_dataloader(opt, train=False)
    denormalizer = Denormalizer(opt)

    # STRIP detector
    strip_detector = STRIP(opt)

    # Entropy list
    list_entropy_trojan = []
    list_entropy_benign = []

    if mode == "attack":
        # Testing with perturbed data
        gauss_smooth = T.GaussianBlur(
            kernel_size=opt.kernel_size, sigma=opt.sigma)
        print("Testing with bd data !!!!")
        inputs, targets = next(iter(test_dataloader))
        inputs = inputs.to(opt.device)
        noise_bd = netG(inputs)
        noise_bd = low_freq(noise_bd, opt)
        bd_inputs = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)
        bd_inputs = gauss_smooth(bd_inputs)
        bd_inputs = denormalizer(bd_inputs) * 255.0
        bd_inputs = bd_inputs.detach().cpu().numpy()
        bd_inputs = np.clip(bd_inputs, 0, 255).astype(
            np.uint8).transpose((0, 2, 3, 1))
        for index in range(opt.n_test):
            background = bd_inputs[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_trojan.append(entropy)
            progress_bar(index, opt.n_test)

        # Testing with clean data
        for index in range(opt.n_test):
            background, _ = testset[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_benign.append(entropy)
    else:
        # Testing with clean data
        print("Testing with clean data !!!!")
        for index in range(opt.n_test):
            background, _ = testset[index]
            entropy = strip_detector(background, testset, netC)
            list_entropy_benign.append(entropy)
            progress_bar(index, opt.n_test)

    return list_entropy_trojan, list_entropy_benign


def main():
    opt = config.get_arguments().parse_args()
    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_workers = 40
        opt.num_classes = 8
    elif opt.dataset == "imagenet10":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
        opt.num_classes = 10
        opt.bs = 32
    else:
        raise Exception("Invalid Dataset")

    if "2" in opt.attack_mode:
        mode = "attack"
    else:
        mode = "clean"
    print(mode)

    lists_entropy_trojan = []
    lists_entropy_benign = []
    for test_round in range(opt.test_rounds):
        list_entropy_trojan, list_entropy_benign = strip(opt, mode)
        lists_entropy_trojan += list_entropy_trojan
        lists_entropy_benign += list_entropy_benign

    # Save result to file
    result_dir = os.path.join(opt.results, opt.dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, "{}_result.txt".format(opt.dataset))

    with open(result_path, "w+") as f:
        for index in range(len(lists_entropy_trojan)):
            if index < len(lists_entropy_trojan) - 1:
                f.write("{} ".format(lists_entropy_trojan[index]))
            else:
                f.write("{}".format(lists_entropy_trojan[index]))

        f.write("\n")

        for index in range(len(lists_entropy_benign)):
            if index < len(lists_entropy_benign) - 1:
                f.write("{} ".format(lists_entropy_benign[index]))
            else:
                f.write("{}".format(lists_entropy_benign[index]))

    min_entropy = min(lists_entropy_trojan + lists_entropy_benign)

    # Determining
    print("Min entropy trojan: {}, Detection boundary: {}".format(
        min_entropy, opt.detection_boundary))
    if min_entropy < opt.detection_boundary:
        print("A backdoored model\n")
    else:
        print("Not a backdoor model\n")


if __name__ == "__main__":
    main()
