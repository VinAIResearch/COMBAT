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
from networks.models import AE, Normalizer, Denormalizer, NetC_MNIST, NetC_MNIST2, NetC_MNIST3
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


def create_bd(inputs, opt):
    sx = 1.05
    sy = 1
    nw = int(inputs.shape[3] * sx)
    nh = int(inputs.shape[2] * sy)
    inputs_bd = fn.center_crop(fn.resize(inputs, (nh, nw)), inputs.shape[2:])
    return inputs_bd
        
        
def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None
    
    if(opt.dataset == 'cifar10'):
        # Model
        netC = PreActResNet18().to(opt.device)
    if(opt.dataset == 'gtsrb'):
        # Model
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if(opt.dataset == 'mnist'):     
        netC = NetC_MNIST3().to(opt.device) #PreActResNet10(n_input=1).to(opt.device) #NetC_MNIST().to(opt.device)

    # Optimizer 
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4, nesterov=True)
      
    # Scheduler 
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    
    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl, mask, pattern, tf_writer, epoch, opt):
    torch.autograd.set_detect_anomaly(True)
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0
    
    total_clean = 0     
    total_bd = 0 
    total_clean_correct = 0
    total_bd_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt)
    
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()
        
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(bs * rate_bd)
        if num_bd < 1:
           continue
        inputs_bd = create_bd(inputs[:num_bd], opt) #+ (pattern[None,:,:,:] - inputs[:num_bd]) * mask[None, None, :,:]
        targets_bd = create_targets_bd(targets[:num_bd], opt)
        
        total_inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        total_preds = netC(total_inputs)

        loss_ce = criterion_CE(total_preds, total_targets)
        if torch.isnan(total_preds).any() or torch.isnan(total_targets).any():
            print(total_preds, total_targets)
        loss = loss_ce 
        loss.backward()
        
        optimizerC.step()
        
        total_sample += bs
        total_loss_ce += loss_ce.detach()
        
        total_clean += bs - num_bd
        total_bd += num_bd
        total_clean_correct += torch.sum(torch.argmax(total_preds[num_bd:], dim=1) == total_targets[num_bd:])
        total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)

        avg_acc_clean = total_clean_correct * 100. / total_clean
        avg_acc_bd = total_bd_correct * 100. / total_bd
        avg_loss_ce = total_loss_ce / total_sample
        progress_bar(batch_idx, len(train_dl), 'CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f}'.format(avg_loss_ce,
                                                                                                            avg_acc_clean,
                                                                                                            avg_acc_bd))

        # Save image for debugging
        if(not batch_idx % 50):
            if(not os.path.exists(opt.temps)):
                create_dir(opt.temps)
            #path = os.path.join(opt.temps, 'backdoor_image.png')
            batch_img = torch.cat([inputs[:num_bd], inputs_bd], dim=2)
            if denormalizer is not None:
                batch_img = denormalizer(batch_img)
            grid = torchvision.utils.make_grid(batch_img, normalize=True)
            
    # for tensorboard
    if(not epoch % 1):
        tf_writer.add_scalars('Clean Accuracy', {'Clean': avg_acc_clean, 'Bd': avg_acc_bd}, epoch)
        tf_writer.add_image('Images', grid, global_step=epoch)
        
    schedulerC.step()        


def eval(netC, optimizerC, schedulerC, test_dl, mask, pattern, best_clean_acc, best_bd_acc, tf_writer, epoch, opt):
    print(" Eval:")
    netC.eval()
    
    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_ae_loss = 0
    
    criterion_BCE = torch.nn.BCELoss()
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs
            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)
            
            inputs_bd = create_bd(inputs, opt) #+ (pattern[None,:,:,:] - inputs) * mask[None, None, :,:]
            targets_bd = create_targets_bd(targets, opt)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100. / total_sample
            acc_bd = total_bd_correct * 100. / total_sample
            
            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(acc_clean, best_clean_acc, acc_bd, best_bd_acc)
            progress_bar(batch_idx, len(test_dl), info_string)
            
    # tensorboard
    if(not epoch % 1):
        tf_writer.add_scalars('Test Accuracy', {'Clean': acc_clean,
                                                 'Bd': acc_bd}, epoch)

    # Save checkpoint 
    if(acc_clean > best_clean_acc):
        print(' Saving...')
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        state_dict = {'netC': netC.state_dict(),
                      'schedulerC': schedulerC.state_dict(),
                      'optimizerC': optimizerC.state_dict(),
                      'best_clean_acc': acc_clean,
                      'best_bd_acc': acc_bd,
                      'epoch_current': epoch,
                      'mask': mask,
                      'pattern': pattern}
        torch.save(state_dict, opt.ckpt_path)
    return best_clean_acc, best_bd_acc
    

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
        opt.num_classes = 43
    elif(opt.dataset == 'mnist'):
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel  = 1
    elif(opt.dataset == 'celeba'):
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.num_workers = 40
    else:
        raise Exception("Invalid Dataset")

    # Dataset 
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)
        
    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)
        
    # Load pretrained model
    mode = opt.saving_prefix
    opt.ckpt_folder = os.path.join(opt.checkpoints, '{}_clean'.format(mode), opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, '{}_{}_clean.pth.tar'.format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, 'log_dir')
    create_dir(opt.log_dir)

    if(opt.continue_training):
        if(os.path.exists(opt.ckpt_path)):
            print('Continue training!!')
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict['netC'])
            optimizerC.load_state_dict(state_dict['optimizerC'])
            schedulerC.load_state_dict(state_dict['schedulerC'])

            best_clean_acc = state_dict['best_clean_acc']
            best_bd_acc = state_dict['best_bd_acc']
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
        train(netC, optimizerC, schedulerC, train_dl, mask, pattern, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc = eval(netC,
                                            optimizerC, 
                                            schedulerC, 
                                            test_dl, 
                                            mask, 
                                            pattern, 
                                            best_clean_acc,
                                            best_bd_acc, tf_writer, epoch, opt)
    
    
if(__name__ == '__main__'):
    main()
