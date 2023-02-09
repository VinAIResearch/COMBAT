"""Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.ind = None

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.ind is not None:
            out += shortcut[:, self.ind, :, :]
        else:
            out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, n_input=3, scaler=1):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(n_input, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.linear = nn.Linear(512 * block.expansion * scaler, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        # out = out.view(out.size(0), -1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


class PreActResNetDropout(PreActResNet):
    def __init__(self, *args, **kwargs):
        super(PreActResNetDropout, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(0.5)

    def eval(self):
        self.train(False)
        self.dropout.train()
        return self

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.dropout(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


input_size2scaler = {32: 1, 64: 4, 224: 49}

def PreActResNet18(num_classes=10, n_input=3, input_size=32):
    scaler = input_size2scaler[input_size]
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, n_input=n_input, scaler=scaler)

def PreActResNetDropout18(num_classes=10, n_input=3, input_size=32):
    scaler = input_size2scaler[input_size]
    return PreActResNetDropout(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, n_input=n_input, scaler=scaler)


def PreActResNet10(num_classes=10, n_input=3, input_size=32):
    scaler = input_size2scaler[input_size]
    return PreActResNet(PreActBlock, [1, 1, 1, 1], num_classes=num_classes, n_input=n_input, scaler=scaler)


def PreActResNet34(num_classes=10, n_input=3, input_size=32):
    scaler = input_size2scaler[input_size]
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes, n_input=n_input, scaler=scaler)


def PreActResNet50(num_classes=10, n_input=3, input_size=32):
    scaler = input_size2scaler[input_size]
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes, n_input=n_input, scaler=scaler)


def PreActResNet101(num_classes=10, n_input=3, input_size=32):
    scaler = input_size2scaler[input_size]
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes=num_classes, n_input=n_input, scaler=scaler)


def PreActResNet152(num_classes=10, n_input=3, input_size=32):
    scaler = input_size2scaler[input_size]
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes=num_classes, n_input=n_input, scaler=scaler)


def test():
    net = PreActResNet18()
    y = net((torch.randn(1, 3, 32, 32)))
    print(y.size())


# test()
