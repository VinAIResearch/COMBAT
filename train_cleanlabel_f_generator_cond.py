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
from classifier_models import PreActResNet18, PreActResNet10
from networks.models import AE, Normalizer, Denormalizer, NetC_MNIST, NetC_MNIST2, NetC_MNIST3, CUnetGeneratorv1
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
import torchvision.transforms.functional as F2
from defenses.frequency_based.model import FrequencyModel, FrequencyModelDropout, FrequencyModelDropoutEnsemble
from classifier_models import VGG, DenseNet121, MobileNetV2, ResNet18
from functools import partial

F_MAPPING_NAMES = {
    "original": FrequencyModel,
    "original_holdout": FrequencyModel,
    "original_dropout": FrequencyModelDropout,
    "original_dropout_ensemble": FrequencyModelDropoutEnsemble,
    "vgg13": partial(VGG, "VGG13"),
    "densenet121": DenseNet121,
    "mobilenetv2": MobileNetV2,
    "resnet18": ResNet18,
}


def dct(x, norm="ortho"):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    #Vc = torch.rfft(v, 1, onesided=False)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def dct_2d(x, norm="ortho"):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def create_dir(path_dir):
    list_subdir = path_dir.strip('.').split('/')
    list_subdir.remove('')
    base_dir = './'
    for subdir in list_subdir:
        base_dir = os.path.join(base_dir, subdir)
        try:
            os.mkdir(base_dir)
        except:
            pass


def create_targets_bd(targets, opt):
    if(opt.attack_mode == 'all2one'):
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif(opt.attack_mode == 'all2all'):
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


#def create_bd(inputs, opt):
#    sx = 1.05
#    sy = 1
#    nw = int(inputs.shape[3] * sx)
#    nh = int(inputs.shape[2] * sy)
#    inputs_bd = fn.center_crop(fn.resize(inputs, (nh, nw)), inputs.shape[2:])
#    return inputs_bd
        
        
def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None
    netG = None
    optimizerG = None
    schedulerG = None
    netF = None
    
    if(opt.dataset == 'cifar10'):
        # Model
        netC = PreActResNet18().to(opt.device)
        netG = CUnetGeneratorv1(opt).to(opt.device)
    if(opt.dataset == 'gtsrb'):
        # Model
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = CUnetGeneratorv1(opt, n_classes=opt.num_classes).to(opt.device)
    if(opt.dataset == 'mnist'):     
        netC = NetC_MNIST3().to(opt.device) #PreActResNet10(n_input=1).to(opt.device) #NetC_MNIST().to(opt.device)
        netG = CUnetGeneratorv1(opt, in_channels=1).to(opt.device)
    if(opt.dataset == 'celeba'):
        netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = CUnetGeneratorv1(opt, n_classes=opt.num_classes).to(opt.device)

    # Optimizer 
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4, nesterov=True)
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    optimizerG = torch.optim.SGD(netG.parameters(), opt.lr_C*0.1, momentum=0.9, weight_decay=5e-4, nesterov=True) #Adam(netG.parameters(), opt.lr_C,betas=(0.9,0.999))
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, opt.schedulerC_milestones, opt.schedulerC_lambda)

    # Frequency Detector
    F_MAPPING_NAMES["original_dropout"] = partial(FrequencyModelDropout, dropout=opt.F_dropout)
    F_MAPPING_NAMES["original_dropout_ensemble"] = partial(FrequencyModelDropoutEnsemble, dropout=opt.F_dropout, num_ensemble=opt.F_num_ensemble)
    netF = F_MAPPING_NAMES[opt.F_model](num_classes=2, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
    netF_eval = F_MAPPING_NAMES[opt.F_model_eval](num_classes=2, n_input=opt.input_channel, input_size=opt.input_height).to(opt.device)
    
    return netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, netF


def train(netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, netF, train_dl, mask, pattern, tf_writer, epoch, opt):
    torch.autograd.set_detect_anomaly(True)
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_loss_grad_l2 = 0
    total_loss_l2 = 0
    total_loss_F = 0
    total_sample = 0
    
    total_clean = 0     
    total_bd = 0 
    total_clean_correct = 0
    total_bd_correct = 0
    total_F_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()
    criterion_L2 = torch.nn.MSELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt)
    k = opt.s * 2 + 1
    
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        ### Train f
        netG.eval()
        netC.train()
        optimizerC.zero_grad()
        # Create backdoor data
        #num_bd = int(bs * rate_bd)
        num_bd = np.sum(np.random.rand(bs) < rate_bd)
        inputs_toChange = inputs[:num_bd]
        noise_bd = netG(inputs_toChange, targets[:num_bd])
        inputs_bd = torch.clamp(inputs_toChange + noise_bd * opt.noise_rate, -1, 1)
        total_inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = targets
        total_preds = netC(total_inputs)

        loss_ce = criterion_CE(total_preds, total_targets)
        if torch.isnan(total_preds).any() or torch.isnan(total_targets).any():
            print(total_preds, total_targets)
        loss = loss_ce
        loss.backward()
        optimizerC.step()


        ### Train G
        netC.eval()
        netG.train()
        optimizerG.zero_grad()
        loss_ce = 0
        loss_l2 = 0
        pred_clean = netC(transforms(inputs))
        total_clean_correct += torch.sum(torch.argmax(pred_clean, dim=1) == targets)
        # Create backdoor data
        ps = int((bs-1)/opt.num_classes) + 1
        inputs_bds = []
        targetss = []
        for ci in range(opt.num_classes):
            si = ci*ps
            ei = si + ps
            if ei > bs: ei = bs
            if si >= ei:
               break
            tmp = targets[si:ei] * 0 + ci
            noise_bd = netG(inputs[si:ei], tmp)
            inputs_bd = torch.clamp(inputs[si:ei] + noise_bd * opt.noise_rate, -1, 1)
            inputs_bds.append(inputs_bd)
            targetss.append(tmp)

        inputs_bd = torch.cat(inputs_bds, 0)
        tmp = torch.cat(targetss, 0)
        pred_bd = netC(transforms(inputs_bd))
        total_bd_correct += torch.sum(torch.argmax(pred_bd, dim=1) == tmp)

        loss_ce += criterion_CE(pred_bd, tmp)
        loss_l2 += criterion_L2(inputs_bd, inputs)
        #inputs_ext = F.pad(inputs, (1,1,2,1))
        #inputs_bd_ext = F.pad(inputs_bd, (1,1,2,1))
        #loss_grad_l2 = criterion_L2(inputs_ext[:,:,1:] - inputs_ext[:,:,:-1], inputs_bd_ext[:,:,1:] - inputs_bd_ext[:,:,:-1]) + \
        #    criterion_L2(inputs_ext[:, :, :, 1:] - inputs_ext[:, :, :, :-1], inputs_bd_ext[:, :, :, 1:] - inputs_bd_ext[:, :, :, :-1])

        # Loss F
        inputs_F = dct_2d((inputs_bd+1)/2*255)
        F_targets = torch.ones_like(targets)
        pred_F = netF(inputs_F)
        loss_F = criterion_CE(pred_F, torch.zeros_like(targets))

        loss = loss_ce + opt.L2_weight * loss_l2 + opt.F_weight * loss_F
        loss.backward()
        optimizerG.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()
        total_loss_l2 += loss_l2.detach()
        total_loss_F += loss_F.detach()
        total_F_correct += torch.sum(torch.argmax(pred_F, dim=1) == F_targets)

        avg_acc_clean = total_clean_correct * 100. / total_sample
        avg_acc_bd = total_bd_correct * 100. / total_sample 
        avg_acc_F = total_F_correct * 100. / total_sample
        avg_loss_ce = total_loss_ce / total_sample
        avg_loss_l2 = total_loss_l2 / total_sample
        #avg_loss_grad_l2 = total_loss_grad_l2 / total_sample
        avg_loss_F = total_loss_F / total_sample
        progress_bar(batch_idx, len(train_dl), 'CE Loss: {:.4f} | L2 Loss: {:.6f} | F Loss: {:.6f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | F Acc: {:.4f}'.format(avg_loss_ce, avg_loss_l2, avg_loss_F,
                                                                                                            avg_acc_clean,
                                                                                                            avg_acc_bd,
                                                                                                            avg_acc_F))

        # Save image for debugging
        if(not batch_idx % 5):
            if(not os.path.exists(opt.temps)):
                create_dir(opt.temps)
            #path = os.path.join(opt.temps, 'backdoor_image.png')
            batch_img = torch.cat([inputs, inputs_bd], dim=2)
            if denormalizer is not None:
                batch_img = denormalizer(batch_img)
            grid = torchvision.utils.make_grid(batch_img, normalize=True)
            
    # for tensorboard
    if(not epoch % 1):
        tf_writer.add_scalars('Clean Accuracy', {'Clean': avg_acc_clean, 'Bd': avg_acc_bd, 'F': avg_acc_F, 'L2 Loss' : avg_loss_l2, 'F Loss': avg_loss_F}, epoch)
        tf_writer.add_image('Images', grid, global_step=epoch)
        
    schedulerC.step()        
    schedulerG.step()


def eval(netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, netF, test_dl, mask, pattern, best_clean_acc, best_bd_acc, best_F_acc, tf_writer, epoch, opt):
    print(" Eval:")
    netC.eval()
    
    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_F_correct = 0
    k = opt.s * 2 + 1

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs
            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            for ci in range(opt.num_classes):
                tmp = targets *0 + ci
                noise_bd = netG(inputs, tmp) #+ (pattern[None,:,:,:] - inputs) * mask[None, None, :,:]
                inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)
                preds_bd = netC(inputs_bd)
                total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == tmp)

                # Evaluate against Frequency Defense
                inputs_F = dct_2d(((inputs_bd+1)/2*255).byte())
                targets_F = torch.ones_like(targets)
                preds_F = netF(inputs_F)
                total_F_correct += torch.sum(torch.argmax(preds_F, 1) == targets_F)

            acc_clean = total_clean_correct * 100. / total_sample
            acc_bd = total_bd_correct * 100. / (total_sample * opt.num_classes)
            acc_F = total_F_correct * 100. / (total_sample * opt.num_classes)
            
            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | F Acc: {:.4f} - Best: {:.4f}".format(acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_F, best_F_acc)
            progress_bar(batch_idx, len(test_dl), info_string)
            
    # tensorboard
    if(not epoch % 1):
        tf_writer.add_scalars('Test Accuracy', {'Clean': acc_clean, 'Bd': acc_bd, 'F': acc_F}, epoch)

    # Save checkpoint 
    if(acc_clean > best_clean_acc):
        print(' Saving...')
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        best_F_acc = acc_F
        state_dict = {'netC': netC.state_dict(),
                      'schedulerC': schedulerC.state_dict(),
                      'optimizerC': optimizerC.state_dict(),
                      'netG': netG.state_dict(),
                      'schedulerG': schedulerG.state_dict(),
                      'optimizerG': optimizerG.state_dict(),
                      'best_clean_acc': acc_clean,
                      'best_bd_acc': acc_bd,
                      'best_F_acc': acc_F,
                      'epoch_current': epoch,
                      'mask': mask,
                      'pattern': pattern}
        torch.save(state_dict, opt.ckpt_path)
    return best_clean_acc, best_bd_acc, best_F_acc
    

def main():
    opt = config.get_arguments().parse_args()
    if(opt.dataset == 'cifar10'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 3 
    elif(opt.dataset == 'gtsrb'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 3
        opt.num_classes = 13
    elif(opt.dataset == 'mnist'):
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel  = 1
    elif(opt.dataset == 'celeba'):
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_workers = 40
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    # Dataset 
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)
        
    # prepare model
    netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, netF = get_model(opt)
        
    # Load pretrained model
    mode = opt.saving_prefix
    opt.ckpt_folder = os.path.join(opt.checkpoints, '{}_clean'.format(mode), opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, '{}_{}_clean.pth.tar'.format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, 'log_dir')
    create_dir(opt.log_dir)

    # Load pretrained FrequencyModel
    opt.F_ckpt_folder = os.path.join(opt.F_checkpoints, opt.dataset, opt.F_model)
    opt.F_ckpt_path = os.path.join(opt.F_ckpt_folder, '{}_{}_detector.pth.tar'.format(opt.dataset, opt.F_model))
    print(f"Loading {opt.F_model} at {opt.F_ckpt_path}")
    state_dict_F = torch.load(opt.F_ckpt_path)
    netF.load_state_dict(state_dict_F['netC'])
    netF.eval()
    print("Done")

    if(opt.continue_training):
        if(os.path.exists(opt.ckpt_path)):
            print('Continue training!!')
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict['netC'])
            optimizerC.load_state_dict(state_dict['optimizerC'])
            schedulerC.load_state_dict(state_dict['schedulerC'])
            netG.load_state_dict(state_dict['netG'])
            optimizerG.load_state_dict(state_dict['optimizerG'])
            schedulerG.load_state_dict(state_dict['schedulerG'])

            best_clean_acc = state_dict['best_clean_acc']
            best_bd_acc = state_dict['best_bd_acc']
            best_F_acc = state_dict['best_F_acc']
            epoch_current = state_dict['epoch_current']

            mask = state_dict['mask']
            pattern = state_dict['pattern']

            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else: 
            print('Pretrained model doesnt exist')
            exit()
    else:
        print('Train from scratch!!!')
        best_clean_acc = 0.
        best_bd_acc = 0.
        best_F_acc = 0.
        epoch_current = 0

        # Prepare mask & pattern
        mask = torch.zeros(opt.input_height, opt.input_width).to(opt.device)
        mask[2:6, 2:6] = 0.1
        pattern = torch.rand(opt.input_channel, opt.input_height, opt.input_width).to(opt.device)
        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        create_dir(opt.log_dir)

        tf_writer = SummaryWriter(log_dir=opt.log_dir)
        
    for epoch in range(epoch_current, opt.n_iters):
        print('Epoch {}:'.format(epoch + 1))
        train(netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, netF, train_dl, mask, pattern, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc, best_F_acc = eval(netC,
                                            optimizerC, 
                                            schedulerC, 
                                            netG,
                                            optimizerG, 
                                            schedulerG, 
                                            netF,
                                            test_dl, 
                                            mask, 
                                            pattern, 
                                            best_clean_acc,
                                            best_bd_acc, best_F_acc, tf_writer, epoch, opt)
    
    
if(__name__ == '__main__'):
    main()
