import torch
import torchvision
import os
import csv
import config
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image


def get_transform(opt, train=True):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    # if(train):
    #     transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.input_height // 8))
    #     transforms_list.append(transforms.RandomRotation(10))
    #     transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transforms_list.append(transforms.ToTensor())
    # if(opt.dataset == 'cifar10'):
    #     transforms_list.append(transforms.Normalize([0.5], [0.25]))
    # elif(opt.dataset == 'mnist'):
    #     transforms_list.append(transforms.Normalize([0.5], [0.5]))
    # elif(opt.dataset == 'gtsrb'):
    #     pass
    # else:
    #     raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        self.num_classes = opt.num_classes
        if(train):
            self.data_folder = os.path.join(opt.data_root, 'GTSRB/Train')
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, 'GTSRB/Test')
            self.images, self.labels = self._get_data_test_list()
            
        self.transforms = transforms
        
    def _get_data_train_list(self):
        images = [] 
        labels = [] 
        l = list(range(0, self.num_classes))
        for c in l:
            prefix = self.data_folder + '/' + format(c, '05d') + '/' 
            gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') 
            gtReader = csv.reader(gtFile, delimiter=';') 
            next(gtReader) 
            for row in gtReader:
                images.append(prefix + row[0]) 
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, 'GT-final_test.csv')
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader)
        l = set(range(0, self.num_classes))
        for row in gtReader:
            if int(row[7]) in l:
                images.append(self.data_folder + '/' + row[0])
                labels.append(int(row[7]))
        return images, labels
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


class CelebA_attr(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type='attr', download=True)
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


def get_dataloader(opt, train=True, shuffle=True):
    transform = get_transform(opt, train)
    if(opt.dataset == 'gtsrb'):
        dataset = GTSRB(opt, train, transform)
    elif(opt.dataset == 'mnist'):
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif(opt.dataset == 'cifar10'):
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    elif(opt.dataset == 'celeba'):
        if(train):
            split = 'train'
        else:
            split = 'test'
        dataset = CelebA_attr(opt, split, transform)
    else:
        raise Exception('Invalid dataset')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=shuffle, pin_memory=True)
    return dataloader


def main():
    opt = config.get_arguments().parse_args()
    transforms = get_transform(opt, False)
    dataloader = get_dataloader(opt, False)
    for item in dataloader:
        images, labels = item
    

if(__name__ == '__main__'):
    main()
