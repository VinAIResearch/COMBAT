import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from classifier_models import PreActResNet18
from config import get_arguments
from networks.models import UnetGenerator
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Function
from utils.dataloader import get_dataloader
from utils.dct import dct_2d, idct_2d


sys.path.insert(0, "../..")


def low_freq(x, opt):
    image_size = opt.input_height
    ratio = opt.ratio
    mask = torch.zeros_like(x)
    mask[:, :, : int(image_size * ratio), : int(image_size * ratio)] = 1
    x_dct = dct_2d((x + 1) / 2 * 255)
    x_dct *= mask
    x_idct = (idct_2d(x_dct) / 255 * 2) - 1
    return x_idct


def text_phantom(text, size):
    # Availability is platform dependent
    font = "LiberationSans-Regular"

    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size // len(text), encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new("RGB", [size, size], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width) // 2, (size - text_height) // 2)
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
    if opt.dataset == "cifar10" or opt.dataset == "gtrsb":
        normalizer = Normalize(opt, [0.5, 0.5], [0.5, 0.5])
    else:
        raise Exception("Invalid dataset")
    return normalizer


def get_denormalize(opt):
    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        denormalizer = Denormalize(opt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    else:
        raise Exception("Invalid dataset")
    return denormalizer


class FeatureExtractor:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

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


class ModelOutputs:
    """Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers."""

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
                x = x.view(x.size(0), -1)
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

        if index is None:
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
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1),
            positive_mask_2,
        )

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
                if module.__class__.__name__ == "ReLU":
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

        if index is None:
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


def get_model(opt):
    print(opt.dataset)
    if opt.dataset == "cifar10":
        # Model
        netC = PreActResNet18().to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    else:
        raise Exception("Invalid dataset")

    # Load pretrained classifier
    path_model = os.path.join(
        opt.checkpoints,
        "{}_clean".format(opt.saving_prefix),
        opt.dataset,
        "{}_{}_clean.pth.tar".format(opt.dataset, opt.saving_prefix),
    )
    state_dict = torch.load(path_model)

    netG.load_state_dict(state_dict["netG"])
    for param in netG.parameters():
        param.requires_grad = False

    netC.load_state_dict(state_dict["netC"])
    for param in netC.parameters():
        param.requires_grad = False
    netC.eval()

    return netC, netG


def get_clean_model(opt):
    if opt.dataset == "cifar10":
        classifier = PreActResNet18()
    else:
        raise Exception("Invalid Dataset")
    # Load pretrained classifier
    clean_model_path = os.path.join(
        opt.checkpoints,
        opt.load_checkpoint_clean,
        opt.dataset,
        "{}_{}.pth.tar".format(opt.dataset, opt.load_checkpoint_clean),
    )
    state_dict = torch.load(clean_model_path)
    classifier.load_state_dict(state_dict["netC"])
    for param in classifier.parameters():
        param.requires_grad = False
    classifier.eval()
    return classifier.to(opt.device)


def show_cam_on_image(img, mask, idx, result_path, opt, prefix=""):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(result_path, prefix + "bd{}.png".format(idx)), np.uint8(img))
    cv2.imwrite(os.path.join(result_path, prefix + "cam{}.png".format(idx)), np.uint8(255 * cam))
    cv2.imwrite("heatmap.png", np.uint8(255 * heatmap))
    heatmap = heatmap[:, :, ::-1].copy()

    heatmap, img = torch.tensor(heatmap).permute(2, 0, 1), torch.tensor(img / 255.0).permute(2, 0, 1)
    heatmap, img = F.interpolate(heatmap.unsqueeze(0), scale_factor=4), F.interpolate(img.unsqueeze(0), scale_factor=4)
    return heatmap[0], img[0]


def create_bd(inputs_clean, generator, opt):
    gauss_smooth = T.GaussianBlur(kernel_size=opt.kernel_size, sigma=opt.sigma)
    noise_bd = generator(inputs_clean)
    noise_bd = low_freq(noise_bd, opt)
    inputs_bd = torch.clamp(inputs_clean + noise_bd * opt.noise_rate, -1, 1)
    inputs_bd = gauss_smooth(inputs_bd)
    targets_bd = torch.ones(inputs_clean.shape[0]).to(opt.device) * opt.target_label

    return inputs_bd, targets_bd


if __name__ == "__main__":
    opt = get_arguments().parse_args()
    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    # args = get_args()

    # Load pretrained model
    model, generator = get_model(opt)
    model_clean = get_clean_model(opt)
    denormalizer = get_denormalize(opt)

    # Prepare dataset
    dl = get_dataloader(opt, False)
    it = iter(dl)
    inputs, targets = next(it)
    inputs, targets = inputs.to(opt.device), targets.to(opt.device)

    # Create backdoor input
    inputs_bd, _ = create_bd(inputs[:20], generator, opt)
    print(inputs_bd.shape)

    grad_cam = GradCam(model=model, feature_module=model.layer3, target_layer_names=["1"], use_cuda=True)
    grad_cam_clean = GradCam(
        model=model_clean, feature_module=model_clean.layer3, target_layer_names=["1"], use_cuda=True
    )
    bs = inputs_bd.shape[0]
    heatmaps = []
    imgs = []
    cams = []

    # for sample_idx in range(bs):
    for idx in range(20):
        input_single = inputs_bd[idx].unsqueeze(0).requires_grad_(True)
        print(input_single.shape)
        if denormalizer:
            img = denormalizer(input_single).squeeze(0)
        else:
            img = input_single.squeeze(0)

        img = img.cpu().detach().numpy() * 255
        print(img.shape)
        img = img.transpose((1, 2, 0))

        result_dir = os.path.join(opt.results, opt.dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None
        mask = grad_cam(input_single, target_index)
        heatmap, img = show_cam_on_image(img, mask, idx, result_dir, opt)
        heatmaps.append(heatmap)
        imgs.append(img)

        input_single = inputs[idx].unsqueeze(0).requires_grad_(True)
        print(input_single.shape)
        if denormalizer:
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
        result_dir = os.path.join(opt.results, opt.dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        heatmap, img = show_cam_on_image(img, mask, idx, result_dir, opt, prefix="clean")
