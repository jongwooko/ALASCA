from __future__ import print_function

import argparse, os, shutil, time, random, math
import numpy as np
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from losses import DCSelect, CTSelect, CTPSelect
from losses.distill import LSLoss, ALSLoss, SDLoss, FDLoss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

def trains(args, trainloader, model, optimizer, criterion, epoch):
    if args.alasca or args.alasca_plus:
        return train_regularize(args, trainloader, model, optimizer, criterion, epoch)
    elif isinstance(criterion, AC_OLSLoss):
        return train_regularize(args, trainloader, model, optimizer, criterion, epoch)
    else:
        return train(args, trainloader, model, optimizer, criterion, epoch)
    
def trains_multi_networks(args, trainloader, model, optimizer, selector, criterion, epoch):
    if args.use_sd or args.alasca or args.alasca_plus:
        return train_multi_networks_regularize(args, trainloader, model, optimizer, selector, criterion, epoch)
    else:
        return train_multi_networks(args, trainloader, model, optimizer, selector, criterion, epoch)
    
def validates(args, valloader, model, criterion):
    if args.use_multi_networks:
        return validate_multi_networks(valloader, model, criterion)
    else:
        return validate(valloader, model, criterion)
    
def train(args, trainloader, model, optimizer, criterion, epoch):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    
    bar = Bar('Training', max=len(trainloader))
    for batch_idx, (inputs, targets, indexs, targets_gt) in enumerate(trainloader):
        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = targets.size(0)
        
        if isinstance(inputs, list):
            inputs_x = inputs[0].cuda()
        else:
            inputs_x = inputs.cuda()
        targets = targets.cuda(non_blocking=True)
        targets_gt = targets_gt.cuda(non_blocking=True)
        
        outputs = model(inputs_x)
        outputs = outputs['outputs'][0]
        loss = criterion(outputs, targets, indexs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), batch_size)
        prec1 = accuracy(outputs, targets, topk=(1,))
        top1.update(prec1[0].item(), batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | ' \
                      'Loss: {loss:.3f} | Prec@1: {top1: .2f} '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    
    return losses.avg, top1.avg

def train_multi_networks(args, trainloader, model, optimizer, selector, criterion, epoch):
    model1, model2 = model
    optimizer1, optimizer2 = optimizer
    
    model1.train()
    model2.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_1 = AverageMeter()
    losses_2 = AverageMeter()
    top1_1 = AverageMeter()
    top1_2 = AverageMeter()
    end = time.time()
    
    num_clean_1, num_clean_2 = 0, 0
    num_total_1, num_total_2 = 0, 0
    
    bar = Bar('Training', max=len(trainloader))
    for batch_idx, (inputs, targets, indexs, targets_gt) in enumerate(trainloader):
        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = targets.size(0)
        
        inputs_x = inputs[0].cuda()
        targets = targets.cuda(non_blocking=True)
        targets_gt = targets_gt.cuda(non_blocking=True)
        
        outputs_1 = model1(inputs_x)
        outputs_2 = model2(inputs_x)
        
        outputs_1 = outputs_1['outputs'][0]
        outputs_2 = outputs_2['outputs'][0]
        
        # Select Instances
        if isinstance(selector, DCSelect):
            ind_1_update, ind_2_update = selector(outputs_1, outputs_2, targets, epoch*len(trainloader)+batch_idx)
        elif isinstance(selector, CTSelect):
            ind_1_update, ind_2_update = selector(outputs_1, outputs_2, targets, epoch)
        elif isinstance(selector, CTPSelect):
            ind_1_update, ind_2_update = selector(outputs_1, outputs_2, targets, epoch, epoch*len(trainloader)+batch_idx)
        
        # Compute Losses
        loss_1 = criterion(outputs_1[ind_2_update], targets[ind_2_update], indexs) * len(ind_2_update) / batch_size
        loss_2 = criterion(outputs_2[ind_1_update], targets[ind_1_update], indexs) * len(ind_1_update) / batch_size
        
        # Backward Step
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        
        losses_1.update(loss_1.item(), batch_size)
        prec1 = accuracy(outputs_1, targets, topk=(1,))
        top1_1.update(prec1[0].item(), batch_size)
        
        losses_2.update(loss_2.item(), batch_size)
        prec1 = accuracy(outputs_2, targets, topk=(1,))
        top1_2.update(prec1[0].item(), batch_size)
        
        num_clean_1 += (targets[ind_2_update] == targets_gt[ind_2_update]).sum().item()
        num_total_1 += len(targets[ind_2_update])
        num_clean_2 += (targets[ind_1_update] == targets_gt[ind_1_update]).sum().item()
        num_total_2 += len(targets[ind_1_update])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | ' \
                      'Net1_Loss: {loss1:.3f} | Net1_Prec@1: {top11: .2f} | ' \
                      'Net2_Loss: {loss2:.3f} | Net2_Prec@1: {top12: .2f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    total=bar.elapsed_td,
                    loss1=losses_1.avg,
                    top11=top1_1.avg,
                    loss2=losses_2.avg,
                    top12=top1_2.avg,
                    )
        bar.next()
    bar.finish()
    return losses_1.avg, losses_2.avg, 0.5 * (num_clean_1 + num_clean_2), 0.5 * (num_total_1 + num_total_2)

def train_regularize(args, trainloader, model, optimizer, criterion, epoch):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    
    total_corr, total_memo, total_inco = 0, 0, 0
    bar = Bar('Training', max=len(trainloader))
    for batch_idx, (inputs, targets, indexs, targets_gt) in enumerate(trainloader):
        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = targets.size(0)
        
        if isinstance(inputs, list):
            inputs_x = inputs[0].cuda()
        else:
            inputs_x = inputs.cuda()
        targets = targets.cuda(non_blocking=True)
        targets_gt = targets_gt.cuda(non_blocking=True)
        
        outputs = model(inputs_x)
        loss = criterion(outputs['outputs'], targets, epoch, indexs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), batch_size)
        prec1 = accuracy(outputs['outputs'][0], targets, topk=(1,))
        top1.update(prec1[0].item(), batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | ' \
                      'Loss: {loss:.3f} | Prec@1: {top1: .2f} '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return losses.avg, top1.avg

def train_byot(args, trainloader, model, optimizer, criterion, epoch):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    
    bar = Bar('Training', max=len(trainloader))
    for batch_idx, (inputs, targets, indexs, targets_gt) in enumerate(trainloader):
        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = targets.size(0)
        
        if isinstance(inputs, list):
            inputs_x = inputs[0].cuda()
        else:
            inputs_x = inputs.cuda()
        targets = targets.cuda(non_blocking=True)
        targets_gt = targets_gt.cuda(non_blocking=True)
        
        outputs = model(inputs_x)
        outs = outputs['outputs']
        feats = outputs['features']
        loss = criterion(outs, feats, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), batch_size)
        prec1 = accuracy(outs[0], targets, topk=(1,))
        top1.update(prec1[0].item(), batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | ' \
                      'Loss: {loss:.3f} | Prec@1: {top1: .2f} '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    
    return losses.avg, top1.avg

def train_multi_networks_regularize(args, trainloader, model, optimizer, selector, criterion, epoch):
    model1, model2 = model
    optimizer1, optimizer2 = optimizer
    
    model1.train()
    model2.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_1 = AverageMeter()
    losses_2 = AverageMeter()
    top1_1 = AverageMeter()
    top1_2 = AverageMeter()
    end = time.time()
    
    num_clean_1, num_clean_2 = 0, 0
    num_total_1, num_total_2 = 0, 0
    
    bar = Bar('Training', max=len(trainloader))
    for batch_idx, (inputs, targets, indexs, targets_gt) in enumerate(trainloader):
        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = targets.size(0)
        
        if isinstance(inputs, list):
            inputs_x = inputs[0].cuda()
        else:
            inputs_x = inputs.cuda()
        targets = targets.cuda(non_blocking=True)
        targets_gt = targets_gt.cuda(non_blocking=True)
        
        outputs_1 = model1(inputs_x)
        outputs_2 = model2(inputs_x)
        
        # Select Instances
        if isinstance(selector, DCSelect):
            ind_1_update, ind_2_update = selector(outputs_1['outputs'][0], outputs_2['outputs'][0], targets, epoch*len(trainloader)+batch_idx)
        elif isinstance(selector, CTSelect):
            ind_1_update, ind_2_update = selector(outputs_1['outputs'][0], outputs_2['outputs'][0], targets, epoch)
        elif isinstance(selector, CTPSelect):
            ind_1_update, ind_2_update = selector(outputs_1['outputs'][0], outputs_2['outputs'][0], targets, epoch, epoch*len(trainloader)+batch_idx)
        
        # Compute Losses
        loss_1 = criterion(outputs_1['outputs'], targets, epoch, indexs, ind_2_update)
        loss_2 = criterion(outputs_2['outputs'], targets, epoch, indexs, ind_1_update)
        
        # Backward Step
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        
        losses_1.update(loss_1.item(), batch_size)
        prec1 = accuracy(outputs_1['outputs'][0], targets_gt, topk=(1,))
        top1_1.update(prec1[0].item(), batch_size)
        
        losses_2.update(loss_2.item(), batch_size)
        prec1 = accuracy(outputs_2['outputs'][0], targets_gt, topk=(1,))
        top1_2.update(prec1[0].item(), batch_size)
        
        num_clean_1 += (targets[ind_2_update] == targets_gt[ind_2_update]).sum().item()
        num_total_1 += len(targets[ind_2_update])
        num_clean_2 += (targets[ind_1_update] == targets_gt[ind_1_update]).sum().item()
        num_total_2 += len(targets[ind_1_update])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | ' \
                      'Net1_Loss: {loss1:.3f} | Net1_Prec@1: {top11: .2f} | ' \
                      'Net2_Loss: {loss2:.3f} | Net2_Prec@1: {top12: .2f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    total=bar.elapsed_td,
                    loss1=losses_1.avg,
                    top11=top1_1.avg,
                    loss2=losses_2.avg,
                    top12=top1_2.avg,
                    )
        bar.next()
    bar.finish()
    return losses_1.avg, losses_2.avg, 0.5 * (num_clean_1 + num_clean_2), 0.5 * (num_total_1 + num_total_2)

def train_ct_clothing1m(args, trainloader, model, optimizer, selector, criterion, epoch):
    model1, model2 = model
    optimizer1, optimizer2 = optimizer
    trainloader1, trainloader2 = trainloader
    
    model1.train()
    model2.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_1 = AverageMeter()
    losses_2 = AverageMeter()
    top1_1 = AverageMeter()
    top1_2 = AverageMeter()
    end = time.time()
    
    num_clean_1, num_clean_2 = 0, 0
    num_total_1, num_total_2 = 0, 0
    
    bar = Bar('Training', max=len(trainloader))
    for batch_idx, (batch1, batch2) in enumerate(zip(trainloader1, trainloader2)):
        # Measure data loading time
        inputs1, targets1, indexs1, targets_gt1 = batch1
        inputs2, targets2, indexs2, targets_gt2 = batch2
        
        data_time.update(time.time() - end)
        batch_size = targets1.size(0)
        
        inputs_x1 = inputs1.cuda()
        inputs_x2 = inputs2.cuda()
        
        targets1 = targets1.cuda(non_blocking=True)
        targets2 = targets2.cuda(non_blocking=True)
        targets = (targets2, targets1)
        
        targets_gt1 = targets_gt1.cuda(non_blocking=True)
        targets_gt2 = targets_gt2.cuda(non_blocking=True)
        
        with torch.no_grad():
            outputs_1 = model1(inputs_x2)
            outputs_2 = model2(inputs_x1)
            # Select Instances
            ind_1_update, ind_2_update = selector(outputs_1['outputs'][0], outputs_2['outputs'][0], targets, epoch)
        
        outputs_1 = model1(inputs_x1)
        outputs_2 = model2(inputs_x2)
        
        # Compute Losses
        loss_1 = criterion(outputs_1['outputs'], targets1, epoch, indexs1, ind_2_update)
        loss_2 = criterion(outputs_2['outputs'], targets2, epoch, indexs2, ind_1_update)
        
        # Backward Step
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        
        losses_1.update(loss_1.item(), batch_size)
        prec1 = accuracy(outputs_1['outputs'][0], targets_gt1, topk=(1,))
        top1_1.update(prec1[0].item(), batch_size)
        
        losses_2.update(loss_2.item(), batch_size)
        prec1 = accuracy(outputs_2['outputs'][0], targets_gt2, topk=(1,))
        top1_2.update(prec1[0].item(), batch_size)
        
        num_clean_1 += (targets1[ind_2_update] == targets_gt1[ind_2_update]).sum().item()
        num_total_1 += len(targets1[ind_2_update])
        num_clean_2 += (targets2[ind_1_update] == targets_gt2[ind_1_update]).sum().item()
        num_total_2 += len(targets2[ind_1_update])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | ' \
                      'Net1_Loss: {loss1:.3f} | Net1_Prec@1: {top11: .2f} | ' \
                      'Net2_Loss: {loss2:.3f} | Net2_Prec@1: {top12: .2f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader1),
                    total=bar.elapsed_td,
                    loss1=losses_1.avg,
                    top11=top1_1.avg,
                    loss2=losses_2.avg,
                    top12=top1_2.avg,
                    )
        bar.next()
    bar.finish()
    return losses_1.avg, losses_2.avg, 0.5 * (num_clean_1 + num_clean_2), 0.5 * (num_total_1 + num_total_2)
    
def validate(valloader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    bar = Bar('Test', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # Measure data loading time
            data_time.update(time.time() - end)
            batch_size = targets.size(0)

            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            if len(outputs['outputs']) == 1:
                outputs = outputs['outputs'][0]
            else:
                outputs = (outputs['outputs'][0] + outputs['outputs'][-1]) / 2
            loss = criterion(outputs, targets)
            losses.update(loss.item(), batch_size)

            prec1 = accuracy(outputs, targets, topk=(1,))
            top1.update(prec1[0].item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Total: {total:} | ' \
                          'Loss: {loss:.3f} | Prec@1: {top1: .2f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        )
            bar.next()
    bar.finish()
    return losses.avg, top1.avg

def validate_multi_networks(valloader, model, criterion):
    model1, model2 = model
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # switch to evaluate mode
    model1.eval()
    model2.eval()
    
    end = time.time()
    bar = Bar('Test', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # Measure data loading time
            data_time.update(time.time() - end)
            batch_size = targets.size(0)
            
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)
            
            # Compute output
            outputs_1 = model1(inputs)
            outputs_2 = model2(inputs)
            outputs = torch.stack([outputs_1['outputs'][0], outputs_2['outputs'][0], outputs_1['outputs'][-1], outputs_2['outputs'][-1]], dim=1).mean(1)
            
            loss = criterion(outputs, targets)
            losses.update(loss.item(), batch_size)

            prec1 = accuracy(outputs, targets, topk=(1,))
            top1.update(prec1[0].item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Total: {total:} | ' \
                          'Loss: {loss:.3f} | Prec@1: {top1: .2f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        )
            bar.next()
    bar.finish()
    return losses.avg, top1.avg