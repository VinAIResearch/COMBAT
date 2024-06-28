import torch
import torch.nn as nn


input_size2scaler = {32: 1, 64: 4}


class FrequencyModel(nn.Module):
    def __init__(self, num_classes=2, n_input=3, input_size=32):
        super(FrequencyModel, self).__init__()
        scaler = input_size2scaler[input_size]

        self.conv1 = nn.Conv2d(n_input, 32, (3, 3), padding="same")
        self.relu1 = nn.ELU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding="same")
        self.relu2 = nn.ELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(32)

        self.maxpool1 = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding="same")
        self.relu3 = nn.ELU(inplace=True)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, (3, 3), padding="same")
        self.relu4 = nn.ELU(inplace=True)
        self.bn4 = nn.BatchNorm2d(64)

        self.maxpool2 = nn.MaxPool2d((2, 2))
        self.dropout2 = nn.Dropout(0.2)

        self.conv5 = nn.Conv2d(64, 128, (3, 3), padding="same")
        self.relu5 = nn.ELU(inplace=True)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding="same")
        self.relu6 = nn.ELU(inplace=True)
        self.bn6 = nn.BatchNorm2d(128)

        self.maxpool3 = nn.MaxPool2d((2, 2))
        self.dropout3 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(2048 * scaler, num_classes)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class FrequencyModelDropout(FrequencyModel):
    def __init__(self, dropout=0.5, *args, **kwargs):
        super(FrequencyModelDropout, self).__init__(*args, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def eval(self):
        self.train(False)
        self.dropout1.train()
        self.dropout2.train()
        self.dropout3.train()
        return self


class FrequencyModelDropoutEnsemble(FrequencyModelDropout):
    def __init__(self, num_ensemble=3, *args, **kwargs):
        super(FrequencyModelDropoutEnsemble, self).__init__(*args, **kwargs)
        self.num_ensemble = num_ensemble

    def forward(self, x):
        outs = []
        for _ in range(self.num_ensemble):
            inp = x
            for module in self.children():
                inp = module(inp)
            outs.append(inp)
        outs = torch.stack(outs, dim=1)
        out = outs.mean(dim=1)
        return out
