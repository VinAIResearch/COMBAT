import os
import random

import kornia.augmentation as A
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    transforms_list.append(transforms.Resize(
        (opt.input_height, opt.input_width)))

    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop(
                (opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(
                transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    if opt.dataset in ["cifar10", "celeba", "imagenet10"]:
        transforms_list.append(
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        )  # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        if opt.post_transform_option != "no_use":
            if not (opt.post_transform_option == "use_modified"):
                self.random_crop = ProbTransform(
                    A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
                )
            self.random_rotation = ProbTransform(
                A.RandomRotation(opt.random_rotation), p=0.5)
            if opt.dataset == "cifar10":
                self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


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


def get_dataloader(opt, train=True, pretensor_transform=False, target_label=None, bs=None, shuffle=True):
    if bs is None:
        bs = opt.bs
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            opt.data_root, train, transform, download=True)
        if target_label is not None:
            pairs = [(x, y) for x, y in zip(dataset.data,
                                            dataset.targets) if int(y) == target_label]
            dataset.data, dataset.targets = [x[0]
                                             for x in pairs], [x[1] for x in pairs]
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
    if opt.debug:
        dataset = torch.utils.data.Subset(
            dataset, range(min(len(dataset), 1000)))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, num_workers=opt.num_workers, shuffle=shuffle, pin_memory=True
    )
    return dataloader


def main():
    pass


if __name__ == "__main__":
    main()
