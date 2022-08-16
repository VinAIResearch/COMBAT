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
    optimizerG = None
    schedulerG = None
    
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

    # Optimizer 
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4, nesterov=True)
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    optimizerG = torch.optim.SGD(netG.parameters(), opt.lr_C*0.1, momentum=0.9, weight_decay=5e-4, nesterov=True) #Adam(netG.parameters(), opt.lr_C,betas=(0.9,0.999))
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, opt.schedulerC_milestones, opt.schedulerC_lambda)
    
    return netC, optimizerC, schedulerC, netG, optimizerG, schedulerG


def train(netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, train_dl, train_dl2, mask, pattern, tf_writer, epoch, opt):
    torch.autograd.set_detect_anomaly(True)
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_loss_grad_l2 = 0
    total_loss_l2 = 0
    total_sample = 0
    
    total_clean = 0     
    total_bd = 0 
    total_clean_correct = 0
    total_cross_correct = 0
    total_bd_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_L2 = torch.nn.MSELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt)

    l = len(train_dl)
    for batch_idx, batch1, batch2 in zip(range(l), train_dl, train_dl2):
        inputs, targets = batch1
        inputs2, targets2 = batch2
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
        bs = inputs.shape[0]
        bd_targets = create_targets_bd(targets, opt)

        ### Train f
        netG.eval()
        netC.train()
        optimizerC.zero_grad()
        # Create backdoor data
        trg_ind = (targets == bd_targets).nonzero()[:,0]
        ntrg_ind = (targets != bd_targets).nonzero()[:,0]
        num_bd = int(trg_ind.shape[0] * rate_bd)
        if num_bd < 1:
           continue
        inputs_toChange = inputs[trg_ind[:num_bd]]
        noise_bd = netG(inputs_toChange)
        inputs_bd = torch.clamp(inputs_toChange + noise_bd * opt.noise_rate, -1, 1)
        total_inputs = torch.cat([inputs_bd, inputs[trg_ind[num_bd:]], inputs[ntrg_ind]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([bd_targets[trg_ind[:num_bd]], targets[trg_ind[num_bd:]], targets[ntrg_ind]], dim=0)
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
        # Create backdoor data
        noise_bd = netG(inputs)
        inputs_bd = torch.clamp(inputs + noise_bd * opt.noise_rate, -1, 1)
        noise_bd2 = netG(inputs2)
        inputs_bd2 = torch.clamp(inputs + noise_bd2 * opt.noise_rate, -1, 1)
        #total_inputs = transforms(total_inputs)
        pred_clean = netC(transforms(inputs))
        pred_cross = netC(transforms(inputs_bd2))
        pred_bd = netC(transforms(inputs_bd))

        loss_ce = criterion_CE(pred_clean, targets) + 10 * criterion_CE(pred_cross, targets) + 50 * criterion_CE(pred_bd, bd_targets)
        if torch.isnan(total_preds).any() or torch.isnan(total_targets).any():
            print(total_preds, total_targets)
        loss_l2 = criterion_L2(inputs_bd, inputs)
        # inputs_ext = F.pad(inputs, (1,1,2,1))
        # inputs_bd_ext = F.pad(inputs_bd, (1,1,2,1))
        # loss_grad_l2 = criterion_L2(inputs_ext[:,:,1:] - inputs_ext[:,:,:-1], inputs_bd_ext[:,:,1:] - inputs_bd_ext[:,:,:-1]) + \
        #         criterion_L2(inputs_ext[:, :, :, 1:] - inputs_ext[:, :, :, :-1], inputs_bd_ext[:, :, :, 1:] - inputs_bd_ext[:, :, :, :-1])

        loss = loss_ce + loss_l2 #+ 10000 * loss_grad_l2 # 0.5*loss_ce + 100*loss_l2
        loss.backward()
        optimizerG.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()
        total_loss_l2 += loss_l2.detach()
        # total_loss_grad_l2 += loss_grad_l2.detach()
        total_clean_correct += torch.sum(torch.argmax(pred_clean, dim=1) == targets)
        total_cross_correct += torch.sum(torch.argmax(pred_cross, dim=1) == targets)
        total_bd_correct += torch.sum(torch.argmax(pred_bd, dim=1) == bd_targets)

        avg_acc_clean = total_clean_correct * 100. / total_sample
        avg_acc_cross = total_cross_correct * 100. / total_sample
        avg_acc_bd = total_bd_correct * 100. / total_sample
        avg_loss_ce = total_loss_ce / total_sample
        avg_loss_l2 = total_loss_l2 / total_sample
        # avg_loss_grad_l2 = total_loss_grad_l2 / total_sample
        progress_bar(batch_idx, len(train_dl), 'CE Loss: {:.4f} | L2 Loss: {:.6f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross Acc: {:.4f}'.format(avg_loss_ce, avg_loss_l2,
                                                                                                            # avg_loss_grad_l2,
                                                                                                            avg_acc_clean,
                                                                                                            avg_acc_bd, avg_acc_cross))

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
        tf_writer.add_scalars('Clean Accuracy', {'Clean': avg_acc_clean, 'Bd': avg_acc_bd, 'Cross': avg_acc_cross, 'L2' : avg_loss_l2}, epoch)
        tf_writer.add_image('Images', grid, global_step=epoch)
        
    schedulerC.step()        


def eval(netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, test_dl, test_dl2, mask, pattern, best_clean_acc, best_bd_acc, best_cross_acc, tf_writer, epoch, opt):
    print(" Eval:")
    netC.eval()
    
    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    
    l = len(test_dl)
    for batch_idx, batch1, batch2 in zip(range(l), test_dl, test_dl2):
        with torch.no_grad():
            inputs, targets = batch1
            inputs2, targets2 = batch2
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

            pred_cross = netC(inputs_bd2)
            total_cross_correct += torch.sum(torch.argmax(pred_cross, 1) == targets)

            acc_clean = total_clean_correct * 100. / total_sample
            acc_cross = total_cross_correct * 100. / total_sample
            acc_bd = total_bd_correct * 100. / total_sample
            
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
                      'schedulerG': schedulerG.state_dict(),
                      'optimizerG': optimizerG.state_dict(),
                      'best_clean_acc': acc_clean,
                      'best_bd_acc': acc_bd,
                      'best_cross_acc': acc_cross,
                      'epoch_current': epoch,
                      'mask': mask,
                      'pattern': pattern}
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
    train_dl2 = get_dataloader(opt, True)
    test_dl2 = get_dataloader(opt, False)
        
    # prepare model
    netC, optimizerC, schedulerC, netG, optimizerG, schedulerG = get_model(opt)
        
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
            netG.load_state_dict(state_dict['netG'])
            optimizerG.load_state_dict(state_dict['optimizerG'])
            schedulerG.load_state_dict(state_dict['schedulerG'])

            best_clean_acc = state_dict['best_clean_acc']
            best_bd_acc = state_dict['best_bd_acc']
            best_cross_acc = state_dict['best_cross_acc']
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
        best_cross_acc = 0.
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
        train(netC, optimizerC, schedulerC, netG, optimizerG, schedulerG, train_dl, train_dl2, mask, pattern, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc, best_cross_acc = eval(netC,
                                            optimizerC, 
                                            schedulerC, 
                                            netG,
                                            optimizerG, 
                                            schedulerG, 
                                            test_dl,
                                            test_dl2,
                                            mask, 
                                            pattern, 
                                            best_clean_acc,
                                            best_bd_acc, best_cross_acc, tf_writer, epoch, opt)
    
    
if(__name__ == '__main__'):
    main()
