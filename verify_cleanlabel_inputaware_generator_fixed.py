import config 
import torchvision 
import torch
import os
import shutil
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as fn

from utils.dataloader_cleanbd import get_dataloader, PostTensorTransform
from utils.utils import progress_bar
from classifier_models import PreActResNet18, PreActResNet10, ResNet18
from networks.models import AE, Normalizer, Denormalizer, NetC_MNIST, NetC_MNIST2, NetC_MNIST3, UnetGenerator
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing


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
    
    if(opt.dataset == 'cifar10'):
        # Model
        netC = PreActResNet18().to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    if(opt.dataset == 'gtsrb'):
        # Model
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)
    if(opt.dataset == 'mnist'):     
        netC = NetC_MNIST3().to(opt.device) #PreActResNet10(n_input=1).to(opt.device) #NetC_MNIST().to(opt.device)
        netG = UnetGenerator(opt, in_channels=1).to(opt.device)
    if(opt.dataset == 'celeba'):
        netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
        netG = UnetGenerator(opt).to(opt.device)

    # Optimizer 
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4, nesterov=True)
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    return netC, optimizerC, schedulerC, netG


def train(netC, optimizerC, schedulerC, netG, train_dl, tf_writer, epoch, opt):
    torch.autograd.set_detect_anomaly(True)
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_loss_l2 = 0
    total_sample = 0
    
    total_clean = 0     
    total_bd = 0 
    total_clean_correct = 0
    total_bd_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()
    criterion_L2 = torch.nn.MSELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt)
    
    for batch_idx, (inputs, targets, poisoned) in enumerate(train_dl):
        inputs, targets, poisoned = inputs.to(opt.device), targets.to(opt.device), poisoned.to(opt.device)
        bs = inputs.shape[0]
        bd_targets = create_targets_bd(targets, opt)

        ### Train f
        #netG.eval()
        netC.train()
        optimizerC.zero_grad()
        # Create backdoor data
        trg_ind = poisoned.nonzero()[:,0]
        ntrg_ind = (poisoned == False).nonzero()[:,0]
        num_bd = trg_ind.shape[0]
        # if num_bd < 1:
        #    continue
        inputs_toChange = inputs[trg_ind]
        noise_bd = netG(inputs_toChange)
        inputs_bd = torch.clamp(inputs_toChange + noise_bd * opt.noise_rate, -1, 1)
        total_inputs = torch.cat([inputs_bd, inputs[ntrg_ind]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([bd_targets[trg_ind], targets[ntrg_ind]], dim=0)
        total_preds = netC(total_inputs)

        loss_ce = criterion_CE(total_preds, total_targets)
        if torch.isnan(total_preds).any() or torch.isnan(total_targets).any():
            print(total_preds, total_targets)
        loss = loss_ce
        loss.backward()
        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()
        total_clean_correct += torch.sum(torch.argmax(total_preds, dim=1) == total_targets)

        avg_acc_clean = total_clean_correct * 100. / total_sample
        avg_loss_ce = total_loss_ce / total_sample
        progress_bar(batch_idx, len(train_dl), 'CE Loss: {:.4f} | Clean Acc: {:.4f}'.format(avg_loss_ce, avg_acc_clean))

        # Save image for debugging
        if(not batch_idx % 5 and num_bd >= 1):
            if(not os.path.exists(opt.temps)):
                create_dir(opt.temps)
            #path = os.path.join(opt.temps, 'backdoor_image.png')
            batch_img = torch.cat([inputs_toChange, inputs_bd], dim=2)
            if denormalizer is not None:
                batch_img = denormalizer(batch_img)
            grid = torchvision.utils.make_grid(batch_img, normalize=True)
            
    # for tensorboard
    if(not epoch % 1):
        tf_writer.add_scalars('Clean Accuracy', {'Clean': avg_acc_clean}, epoch)
        tf_writer.add_image('Images', grid, global_step=epoch)
        
    schedulerC.step()        


def eval(netC, optimizerC, schedulerC, netG, test_dl, test_dl2, best_clean_acc, best_bd_acc, best_cross_acc, tf_writer, epoch, opt):
    print(" Eval:")
    netC.eval()
    
    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_ae_loss = 0
    
    l = len(test_dl)
    for batch_idx, batch1, batch2 in zip(range(l), test_dl, test_dl2):
        with torch.no_grad():
            inputs, targets, _ = batch1
            inputs2, targets2, _ = batch2
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs
            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            noise_bd = netG(inputs)
            inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)
            noise_bd2 = netG(inputs2)
            inputs_bd2 = torch.clamp(inputs + noise_bd2 * opt.noise_rate, -1, 1)
            targets_bd = create_targets_bd(targets, opt)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            preds_cross = netC(inputs_bd2)
            total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

            acc_clean = total_clean_correct * 100. / total_sample
            acc_bd = total_bd_correct * 100. / total_sample
            acc_cross = total_cross_correct * 100. / total_sample
            
            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | Cross Acc: {:.4f} - Best: {:.4f}".format(acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_cross, best_cross_acc)
            progress_bar(batch_idx, len(test_dl), info_string)
            
    # tensorboard
    if(not epoch % 1):
        tf_writer.add_scalars('Test Accuracy', {'Clean': acc_clean,
                                                 'Bd': acc_bd, 'Cross': acc_cross}, epoch)

    # Save checkpoint 
    if(acc_clean > best_clean_acc):
        print(' Saving...')
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        best_cross_acc = acc_cross
        state_dict = {'netC': netC.state_dict(),
                      'schedulerC': schedulerC.state_dict(),
                      'optimizerC': optimizerC.state_dict(),
                      'netG': netG.state_dict(),
                      'best_clean_acc': acc_clean,
                      'best_bd_acc': acc_bd,
                      'best_cross_acc': acc_cross,
                      'epoch_current': epoch}
        torch.save(state_dict, opt.ckpt_path)
    return best_clean_acc, best_bd_acc, best_cross_acc
    

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
    test_dl2 = get_dataloader(opt, False)
        
    # prepare model
    netC, optimizerC, schedulerC, netG = get_model(opt)
        
    # Load pretrained model
    mode = opt.saving_prefix
    opt.ckpt_folder = os.path.join(opt.checkpoints, '{}_clean'.format(mode), opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, '{}_{}_clean.pth.tar'.format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, 'log_dir')
    create_dir(opt.log_dir)

    load_path = os.path.join(opt.checkpoints, opt.load_checkpoint, opt.dataset, '{}_{}.pth.tar'.format(opt.dataset, opt.load_checkpoint))
    if(not os.path.exists(load_path)):
            print('Error: {} not found'.format(load_path))
            exit()
    else:
            state_dict = torch.load(load_path)
            netG.load_state_dict(state_dict['netG'])
            netG.eval()
            tf_writer = SummaryWriter(log_dir=opt.log_dir)
    best_clean_acc = 0.
    best_bd_acc = 0.
    best_cross_acc = 0.
    epoch_current = 0        
    for epoch in range(epoch_current, opt.n_iters):
        print('Epoch {}:'.format(epoch + 1))
        train(netC, optimizerC, schedulerC, netG, train_dl, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc, best_cross_acc = eval(netC,
                                            optimizerC, 
                                            schedulerC, 
                                            netG, 
                                            test_dl,
                                            test_dl2,
                                            best_clean_acc,
                                            best_bd_acc, best_cross_acc, tf_writer, epoch, opt)
    
    
if(__name__ == '__main__'):
    main()
