import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_arguments

sys.path.insert(0, "../..")
from classifier_models import ResNet18
from networks.models import AE, Denormalizer, Normalizer, UnetGenerator
from utils.dataloader import get_dataloader
from utils.utils import progress_bar


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def convert(mask):
    mask_len = len(mask)
    converted_mask = torch.ones(mask_len * 4, dtype=bool)
    for i in range(4):
        for j in range(mask_len):
            try:
                converted_mask[4 * j + i] = mask[j]
            except:
                print(i, j)
                input()
    return converted_mask


def eval(netC, netG, test_dl, opt):
    print(" Eval:")
    acc_clean = 0.0
    acc_bd = 0.0
    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs
            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            noise_bd = netG(inputs)  # + (pattern[None,:,:,:] - inputs) * mask[None, None, :,:]
            inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)
            targets_bd = create_targets_bd(targets, opt)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

        progress_bar(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))
    return acc_clean, acc_bd


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()

    if opt.dataset == "celeba":
        opt.num_classes = 8
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    else:
        raise Exception("Invalid Dataset")

    mode = opt.saving_prefix
    path_model = os.path.join(opt.checkpoints, "{}_clean".format(opt.saving_prefix), opt.dataset, "{}_{}_clean.pth.tar".format(opt.dataset, opt.saving_prefix))
    state_dict = torch.load(path_model)
    print("load G")
    netG.load_state_dict(state_dict["netG"])
    netG.to(opt.device)
    print("load C")
    netC.load_state_dict(state_dict["netC"])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    print(state_dict["best_clean_acc"], state_dict["best_bd_acc"])

    # Prepare dataloader
    test_dl = get_dataloader(opt, train=False)

    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    hook = netC.layer4.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(test_dl):
        print(batch_idx)
        inputs = inputs.to(opt.device)
        netC(inputs)
        progress_bar(batch_idx, len(test_dl))

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    with open(opt.outfile, "w") as outs:
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(netC)
            num_pruned = index
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            print("Pruned {} filters".format(num_pruned))

            net_pruned.layer4[1].conv2 = nn.Conv2d(pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False)
            net_pruned.linear = nn.Linear(4 * (pruning_mask.shape[0] - num_pruned), 8)

            # Re-assigning weight to the pruned net
            for name, module in net_pruned._modules.items():
                if "layer4" in name:
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].bn2.running_mean = netC.layer4[1].bn2.running_mean[pruning_mask]
                    module[1].bn2.running_var = netC.layer4[1].bn2.running_var[pruning_mask]
                    module[1].bn2.weight.data = netC.layer4[1].bn2.weight.data[pruning_mask]
                    module[1].bn2.bias.data = netC.layer4[1].bn2.bias.data[pruning_mask]

                    module[1].ind = pruning_mask

                elif "linear" == name:
                    converted_mask = convert(pruning_mask)
                    module.weight.data = netC.linear.weight.data[:, converted_mask]
                    module.bias.data = netC.linear.bias.data
                else:
                    continue
            net_pruned.to(opt.device)
            clean, bd = eval(net_pruned, netG, test_dl, opt)
            outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))


if __name__ == "__main__":
    main()
