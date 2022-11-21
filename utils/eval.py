from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
import torch.optim as optim

import time
from progress.bar import Bar as Bar
from .misc import AverageMeter
from .logger import Logger

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    if output.size(0) == batch_size:
        _, pred = output.topk(maxk, 1, True, True)
        
    elif output.size(0) % batch_size == 0:
        pred = torch.zeros((batch_size, output.size(1))).cuda()
        for i in range(int(output.size(0) / batch_size)):
            preds = nn.functional.softmax(output[batch_size*i:batch_size*(i+1)], dim=-1)
            pred += preds
        pred /= output.size(0) / batch_size
        _, pred = pred.topk(maxk, 1, True, True)
    else:
        print (output.size(0), batch_size)
        raise NotImplementedError
        
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res