import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, targets, alpha):
        _targets = targets
        targets = torch.zeros(outputs.size(), device=targets.device).scatter_(1, _targets.view(-1,1), 1)
        all_one = torch.ones(outputs.size(), device=targets.device).scatter_(1, _targets.view(-1,1), 0)
        
        targets = alpha * targets
        targets += (1 - alpha) / (outputs.size(1) - 1) * all_one
        loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1))
        return loss   

class ALASCA(nn.Module):
    """
    Adaptive LAbel Smoothing on auxiliary ClAssifier
    """
    def __init__(self, criterion, num_examp, num_class, lam=2.0, w_ema=0.7, temp=0.33):
        super().__init__()
        self.criterion = criterion
        self.middle_loss_fn = NLLLoss()
        self.lam = lam
        self.w_ema = w_ema
        self.temp = temp
        self.ema = torch.zeros((num_examp, num_class)).cuda() / num_class
        self.warmup_epoch = 30
    
    def forward(self, outputs, targets, epoch, indexs, update=None):
        _targets = targets
        
        if update is None:
            loss = self.criterion(outputs[0], targets, indexs)
        else:
            loss = self.criterion(outputs[0][update], targets[update], indexs[update])
        self.ema[indexs] = self.w_ema * self.ema[indexs] + (1 - self.w_ema) * outputs[0].detach()
        ema_sm = F.softmax(self.ema[indexs] / self.temp, dim=-1).detach()
        alpha = torch.gather(ema_sm, 1, _targets.view(-1, 1))
        alpha = min(1., epoch / self.warmup_epoch) * alpha + (1 - min(1., epoch / self.warmup_epoch)) * torch.ones(alpha.size()).cuda()
        for i in range(1, len(outputs)):
            loss += self.lam * self.middle_loss_fn(outputs[i], _targets, alpha.view(-1, 1))
        return loss
    
class clothing1M_ALASCA(nn.Module):
    """
    Adaptive LAbel Smoothing on auxiliary ClAssifier
    """
    def __init__(self, criterion, num_examp, num_class, lam=1.0, w_ema=0.7, temp=3.0):
        super().__init__()
        self.criterion = criterion
        self.middle_loss_fn = NLLLoss()
        self.lam = lam
        self.w_ema = w_ema
        self.temp = temp
        self.ema = torch.zeros((num_examp, num_class)).cuda() / num_class
        self.warmup_epoch = 3
    
    def forward(self, outputs, targets, epoch, indexs, update=None):
        _targets = targets
        
        if update is None:
            loss = self.criterion(outputs[0], targets, indexs)
        else:
            loss = self.criterion(outputs[0][update], targets[update], indexs[update])
        ema_sm = F.softmax(outputs[0].detach(), dim=-1).detach()
        alpha = torch.gather(ema_sm, 1, _targets.view(-1, 1))
        alpha = min(1., epoch / self.warmup_epoch) * alpha + (1 - min(1., epoch / self.warmup_epoch)) * torch.ones(alpha.size()).cuda()
        for i in range(1, len(outputs)):
            loss += self.lam * self.middle_loss_fn(outputs[i], _targets, alpha.view(-1, 1))
        return loss