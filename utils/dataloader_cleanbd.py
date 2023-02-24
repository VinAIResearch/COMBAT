import csv
import os
import random
import glob

import kornia.augmentation as A
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Dataset classes with pre-computed and fixed image indices for poisoning. Defined by the boolean list self.poisoned


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
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))

    if opt.dataset == 'tinyimagenet':
        transforms_list.append(transforms.Lambda(lambda x: x.convert('RGB')))

    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10" or opt.dataset == 'tinyimagenet':
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))  # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset in ['gtsrb', 'gtsrb2', 'celeba', 'imagenet10', 'tinyimagenet']:
        transforms_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))  # pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        if opt.post_transform_option != "no_use":
            if not (opt.dataset != "gtsrb" and opt.post_transform_option == "use_modified"):
                self.random_crop = ProbTransform(A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8)
            self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
            if opt.dataset == "cifar10":
                self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        assert opt.target_label < opt.num_classes
        self.num_classes = opt.num_classes
        if opt.attack_mode == "all2one":
            target_label = {opt.target_label}
        else:
            target_label = set(range(0, opt.num_classes))
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels, self.poisoned = self._get_data_train_list(target_label, opt.pc)
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels, self.poisoned = self._get_data_test_list(target_label, opt.pc)
        self.transforms = transforms

    def _get_data_train_list(self, target_label, pc):
        images = []
        labels = []
        poisoned = []
        l = list(range(self.num_classes))
        for c in l:
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
                if c in target_label:  # Define poisoning status
                    if random.random() < pc:
                        poisoned.append(True)
                    else:
                        poisoned.append(False)
                else:
                    poisoned.append(False)
            gtFile.close()
        return images, labels, poisoned

    def _get_data_test_list(self, target_label, pc):
        images = []
        labels = []
        poisoned = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        l = set(range(self.num_classes))
        for row in gtReader:
            if int(row[7]) in l:
                images.append(self.data_folder + "/" + row[0])
                labels.append(int(row[7]))
                if int(row[7]) in target_label:  # Define poisoning status
                    if random.random() < pc:
                        poisoned.append(True)
                    else:
                        poisoned.append(False)
                else:
                    poisoned.append(False)

        return images, labels, poisoned

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        poisoned = self.poisoned[index]
        return image, label, poisoned


class GTSRB2(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB2, self).__init__()
        if opt.attack_mode == "all2one":
            target_label = {opt.target_label}
        else:
            target_label = set(range(0, 43))
        self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
        self.images, self.labels, self.poisoned = self._get_data_train_list(target_label, opt.pc)
        self.transforms = transforms

    def _get_data_train_list(self, target_label, pc):
        images = []
        labels = []
        poisoned = []
        l = list(range(0, 43))
        if target_label is not None:
            l = [target_label]
        for c in l:
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                res = int(row[0][:-4][-5:])
                if res > 27:
                    images.append(prefix + row[0])
                    labels.append(int(row[7]))
                    if int(row[7]) in target_label:  # Define poisoning status
                        if random.random() < pc:
                            poisoned.append(True)
                        else:
                            poisoned.append(False)
                    else:
                        poisoned.append(False)
            gtFile.close()
        return images, labels, poisoned

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        poisoned = self.poisoned[index]
        return image, label, poisoned


class CelebA_attr(data.Dataset):  # Have not  updated
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type="attr", download=True)
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
        self.dataset = torchvision.datasets.ImageNet(root=os.path.join(opt.data_root, "imagenet10"), split=split)
        self.transforms = transforms
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        return (input, target)


class TinyImageNet(data.Dataset):
    def __init__(self, opt, split, transform=None):
        #self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.split_dir = os.path.join(opt.data_root, 'TinyImageNet', self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % 'JPEG'), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing
        with open(os.path.join(opt.data_root, 'TinyImageNet/wnids.txt'), 'r') as fp:
                self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}
        
        # Text label - number mapping 
        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(500):
                    self.labels['%s_%d.%s' % (label_text, cnt, 'JPEG')] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, 'val_annotations.txt'), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img
    
    def __getitem__(self, index):
        file_path = self.image_paths[index]
        img = self.read_image(file_path)

        # if self.split == 'test':
        #     return img
        #
            # file_name = file_path.split('/')[-1]
        return img, self.labels[os.path.basename(file_path)]


# class PoisonedDataset(data.Dataset):
#     def __init__(self, refdata, n_classes, opt):
#         self.dataset = refdata
#         if opt.attack_mode == 'all2one':
#             target_label = {opt.target_label}
#         else:
#             target_label = set(range(0,n_classes))
#         self.poisoned = self._poison_flags(target_label, opt.pc)
#
#     def _poison_flags(self, target_label, pc):
#         poisoned = []
#         for _, label in tqdm(self.dataset, desc="Define poisoning status"):
#             if int(label) in target_label:          # Define poisoning status
#                 if random.random() < pc:
#                     poisoned.append(True)
#                 else:
#                     poisoned.append(False)
#             else:
#                 poisoned.append(False)
#         return poisoned
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         input, target = self.dataset[index]
#         poisoned = self.poisoned[index]
#         return (input, target, poisoned)


class PoisonedDataset(data.Dataset):
    def __init__(self, refdata, n_classes, opt):
        self.dataset = refdata
        if opt.debug:
            self.dataset = torch.utils.data.Subset(self.dataset, range(min(len(self.dataset), 1000)))
        if opt.attack_mode == "all2one":
            target_label = {opt.target_label}
        else:
            target_label = set(range(0, n_classes))
        self.poisoned = self._poison_flags(target_label, opt.pc)

    def _poison_flags(self, target_label, pc):
        targeted_image_ids = []
        for idx, (_, label) in enumerate(tqdm(self.dataset, desc="Define poisoning status")):
            if int(label) in target_label:  # Define poisoning status
                targeted_image_ids.append(idx)
        num_poisoned = max(0, int(pc * len(targeted_image_ids)))
        print(f"Poison {num_poisoned} images ({pc * len(targeted_image_ids)})")
        poisoned = set(random.sample(targeted_image_ids, num_poisoned))
        return poisoned

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        poisoned = index in self.poisoned
        return (input, target, poisoned)


def get_dataloader(opt, train=True, pretensor_transform=False, bs=None, shuffle=True):
    if bs is None:
        bs = opt.bs
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "gtsrb2":
        dataset = GTSRB2(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = PoisonedDataset(torchvision.datasets.MNIST(opt.data_root, train, transform, download=True), opt.num_classes, opt)
    elif opt.dataset == "cifar10":
        dataset = PoisonedDataset(torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True), opt.num_classes, opt)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = PoisonedDataset(CelebA_attr(opt, split, transform), opt.num_classes, opt)
    elif opt.dataset == "imagenet10":
        split = 'train' if train else 'val'
        dataset = PoisonedDataset(ImageNet(opt, split, transform), opt.num_classes, opt)
    elif opt.dataset == 'tinyimagenet':
        split = 'train' if train else 'val'
        dataset = PoisonedDataset(TinyImageNet(opt, split, transform), opt.num_classes, opt)
    else:
        raise Exception("Invalid dataset")
    if opt.debug:
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), 1000)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=opt.num_workers, shuffle=shuffle, pin_memory=True)
    return dataloader


def main():
    # opt = config.get_arguments().parse_args()
    # transforms = get_transform(opt, False)
    # dataloader = get_dataloader(opt, False)
    # for item in dataloader:
    #     images, labels = item
    # opt = config.get_arguments().parse_args()
    # dataset = torchvision.datasets.CelebA(root=opt.data_root, split='test',
    #                                       target_type='identity', download=True)
    # sooka = dataset[51439]
    pass


if __name__ == "__main__":
    main()
