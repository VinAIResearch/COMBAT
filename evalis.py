import config
import torchvision
import torch
import os
import shutil
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as fn

from utils.dataloader import get_dataloader, PostTensorTransform
from utils.utils import progress_bar
from classifier_models import PreActResNet18, PreActResNet10, ResNet18
from networks.models import AE, Normalizer, Denormalizer, NetC_MNIST, NetC_MNIST2, NetC_MNIST3, UnetGenerator
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from torchvision.transforms.functional import pil_to_tensor

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from glob import glob
from PIL import Image


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.cln_img_dir = os.path.join(img_dir, "cln")
        self.bd_img_dir = os.path.join(img_dir, "bd")
        self.cln_img_paths = os.listdir(self.cln_img_dir)
        self.bd_img_paths = os.listdir(self.bd_img_dir)
        assert len(self.cln_img_paths) == len(self.bd_img_paths), f"Number of clean ({len(self.cln_img_paths)}) and backdoored ({len(self.bd_img_paths)}) images do not match"

    def __len__(self):
        return len(self.cln_img_paths)

    def __getitem__(self, index):
        cln_img_path = os.path.join(self.cln_img_dir, self.cln_img_paths[index])
        bd_img_path = os.path.join(self.bd_img_dir, self.bd_img_paths[index])

        cln = pil_to_tensor(Image.open(cln_img_path)).float()
        bd = pil_to_tensor(Image.open(bd_img_path)).float()

        return cln, bd


def eval(test_dl, opt):
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarityIndexMeasure()
    lpips = LearnedPerceptualImagePatchSimilarity()

    for batch_idx, (img1, img2) in enumerate(tqdm(test_dl)):
        with torch.no_grad():
            psnr.update(img2, img1)
            ssim.update(img2, img1)
            lpips.update((img2/127.5)-1, (img1/127.5)-1)

    psnr_score = psnr.compute()
    ssim_score = ssim.compute()
    lpips_score = lpips.compute()

    print(f"psnr_score = {psnr_score}")
    print(f"ssim_score = {ssim_score}")
    print(f"lpips_score = {lpips_score}")


def main():
    parser = config.get_arguments()
    parser.add_argument("--img_dir", type=str)
    opt = parser.parse_args()


    print(f"Evaluating image quality: {opt.img_dir}")
    test_dl = torch.utils.data.DataLoader(ImgDataset(opt.img_dir), batch_size=opt.bs, shuffle=False)
    eval(test_dl, opt)


if(__name__ == '__main__'):
    main()
