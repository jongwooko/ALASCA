from __future__ import print_function

import argparse
import os, shutil, time, random, math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils import set_seed
from scipy import optimize
from losses import GCELoss, SCELoss, ELRLoss
from losses import SDLoss, LSLoss, ALSLoss, FDLoss
from losses import DCSelect, CTSelect, CTPSelect
from training_functions import trains, train_ct_clothing1m, validates
from common import save_checkpoint, load_checkpoint, adjust_learning_rate
from arguments import parse_args

args = parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0 # best test accuracy
num_class = 14

import dataset.clothing1M as dataset
    
def main():
    global best_acc
    
    if not os.path.isdir(args.out):
        mkdir_p(args.out)
        
    # Data
    print('==> Preparing Clothing1M Testset')
    
    set_seed(args.manualSeed)
    test_set = dataset.get_clothing1M('/home/work/KAIST-OSI-JONGWOO/dataset/clothing1m/', batch_size=0, num_batches=0, train=False)
    test_loader = data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    
    # Model
    print ("==> creating ResNet-50")
    
    def create_model(args):
        model = models.sdresnet50(num_class)
        model = model.cuda()
        return model
    
    model = create_model(args)
    if args.resume:
        import pandas as pd
        print ("==> Resuming from checkpoint..")
        model = load_checkpoint(model, args.resume, 'checkpoint.pth.tar')
        
    cudnn.benchmark = True
    print ('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    if args.loss_fn == "ce":
        from losses import CrossEntropyLoss
        train_criterion = CrossEntropyLoss()
    elif args.loss_fn == "ls":
        from losses import LabelSmoothingLoss
        train_criterion = LabelSmoothingLoss(alpha=args.alpha)
    elif args.loss_fn == "als":
        from losses import AdaptiveLabelSmoothingLoss
        train_criterion = AdaptiveLabelSmoothingLoss()
    elif args.loss_fn == "gce":
        train_criterion = GCELoss(num_classes=num_class)
    elif args.loss_fn == 'sce':
        train_criterion = SCELoss(num_classes=num_class, a=1.0, b=0.4)
    elif args.loss_fn == 'elr':
        train_criterion = ELRLoss(len(train_labeled_set), num_class=num_class,
                                  beta=args.beta, lmbda=args.lmbda)
    else:
        raise NotImplementedError
        
    if args.alasca:
        from losses import clothing1M_ALASCA
        train_criterion = clothing1M_ALASCA(criterion=train_criterion, num_examp=0, 
                                 num_class=num_class, lam=args.lam)
        
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.01)
    start_epoch = 0
    
    if args.use_multi_networks:
        assert args.loss_fn == "ce"
        if args.multi_networks_method == "coteach":
            selector = CTSelect(forget_rate=args.r, num_gradual=args.num_gradual, n_epoch=args.epochs)
        
        model2 = create_model(args)
        
        optimizer1 = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
        lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[7], gamma=0.01)
        optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
        lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[7], gamma=0.01)
        
        model = [model, model2]
        optimizer = [optimizer1, optimizer2]
    
    # Resume
    title = 'Training Clothing1M dataset'
    logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
    if not args.use_multi_networks:
        logger.set_names(['Loss', 'Test Loss', 'Train@1', 'Prec@1'])
    else:
        logger.set_names(['Loss', 'Test Loss', 'Prec@1', 'Num Clean', 'Num Total'])
    
    test_accs = []
    
    lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        
        print('==> Preparing Clothing1M Trainset')
        if not args.use_multi_networks:
            train_set, _ = dataset.get_clothing1M('/home/work/KAIST-OSI-JONGWOO/dataset/clothing1m/', batch_size=64, num_batches=2000, train=True)
            train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
        else:
            train_set1, _ = dataset.get_clothing1M('/home/work/KAIST-OSI-JONGWOO/dataset/clothing1m/', batch_size=64, num_batches=3000, train=True)
            train_set2, _ = dataset.get_clothing1M('/home/work/KAIST-OSI-JONGWOO/dataset/clothing1m/', batch_size=64, num_batches=3000, train=True)
            train_loader1 = data.DataLoader(train_set1, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
            train_loader2 = data.DataLoader(train_set2, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
            train_loader = [train_loader1, train_loader2]
        
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        
        # Training part
        if not args.use_multi_networks:
            train_loss, train1 = trains(args, train_loader, model, optimizer, train_criterion, epoch)
        else:
            train_loss1, train_loss2, num_clean, num_total = train_ct_clothing1m(args, train_loader, model, optimizer, selector, train_criterion, epoch)
            train_loss = (train_loss1 + train_loss2) / 2.
        
        if isinstance(optimizer, list):
            lr_scheduler1.step()
            lr_scheduler2.step()
            lr = optimizer1.param_groups[0]['lr']
#             for _opt in optimizer:
#                 lr = adjust_learning_rate(_opt, epoch, args)
        else:
            lr_scheduler.step()
            lr = optimizer.param_groups[0]['lr']
        
        # Evaluation part
        test_loss, prec1 = validates(args, test_loader, model, criterion)
        
        # Append logger file
        if not args.use_multi_networks:
            logger.append([train_loss, test_loss, train1, prec1])
        else:
            logger.append([train_loss, test_loss, prec1, num_clean, num_total])
        
        # Save models
        if isinstance(model, list):
            _model = model[0]
        else:
            _model = model
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': _model.state_dict(),
        }, epoch + 1, args.out)
        test_accs.append(prec1)

    logger.close()
    
    # Pring the final results
    print('Mean Acc:')
    print(np.mean(test_accs[-20:]))
    
    print ('Max Acc:')
    print(np.max(test_accs))

    print('Name of saved folder:')
    print(args.out)
    
    
if __name__ == '__main__':
    main()