import os

import config
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


def get_transform(opt, train=True):
    transforms_list = []
    transforms_list.append(transforms.Resize(
        (opt.input_height, opt.input_width)))
    if train:
        transforms_list.append(
            transforms.RandomCrop(
                (opt.input_height, opt.input_width), padding=opt.input_height // 8)
        )
        transforms_list.append(transforms.RandomRotation(10))
        transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    return transforms.Compose(transforms_list)


class CelebA_attr(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(
            root=opt.data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)


class ImageNet(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.ImageNet(
            root=os.path.join(opt.data_root, "imagenet10"), split=split)
        self.transforms = transforms
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        return (input, target)


def get_dataloader(opt, train=True):
    transform = get_transform(opt, train)
    if opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            opt.data_root, train, transform, download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)
    elif opt.dataset == "imagenet10":
        split = "train" if train else "val"
        dataset = ImageNet(opt, split, transform)
    else:
        raise Exception("Invalid dataset")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    return dataloader


def get_dataset(opt, train=True):
    if opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(
            opt,
            split,
            transforms=transforms.Compose(
                [transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    elif opt.dataset == "imagenet10":
        split = "train" if train else "val"
        dataset = ImageNet(
            opt,
            split,
            transforms=transforms.Compose(
                [transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    else:
        raise Exception("Invalid dataset")
    return dataset


def main():
    opt = config.get_arguments().parse_args()
    get_transform(opt, False)
    dataloader = get_dataloader(opt, False)
    for item in dataloader:
        images, labels = item


if __name__ == "__main__":
    main()
