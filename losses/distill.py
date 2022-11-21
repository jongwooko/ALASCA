import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class LabelSmoothingLoss(nn.Module):
    """ Label Smoothing Loss for Final Classifier. """
    def __init__(self, alpha):
        super(LabelSmoothingLoss, self).__init__()
        self.alpha = 1 - alpha
        self.loss_fn = NLLLoss()
        
    def forward(self, outputs, targets, indexs):
        alpha = self.alpha * torch.ones(outputs.size(0), device=targets.device)
        loss = self.loss_fn(outputs, targets, alpha.view(-1,1))
        return loss
    
class AdaptiveLabelSmoothingLoss(nn.Module):
    """ Adaptive Label Smoothing for Final Classifier. """
    def __init__(self):
        super(AdaptiveLabelSmoothingLoss, self).__init__()
        self.loss_fn = NLLLoss()
        self.warmup = 60
        self.ema = torch.zeros((num_examp, num_class)).cuda() / num_class
        
    def forward(self, outputs, targets, epoch):
        alpha = torch.gather(F.softmax(outputs, dim=-1).detach(), 1, targets.view(-1, 1))
        weight = min(1., epoch/self.warmup)
        alpha = 0.5 * (weight * alpha + (1 - weight) * torch.ones(alpha.size()).cuda())
        alpha = 0.5 * torch.ones(alpha.size()).cuda()
        loss = self.loss_fn(outputs, targets, alpha.view(-1, 1))
        return loss
    
class EMA_ALS(nn.Module):
    """ Adaptive Label Smoothing for Final Classifier. """
    def __init__(self, num_examp, num_class):
        super(EMA_ALS, self).__init__()
        self.loss_fn = NLLLoss()
        self.warmup = 60
        self.ema = torch.zeros((num_examp, num_class)).cuda() / num_class
        self.w_ema = 0.7
        self.temp = 3.0
        
    def forward(self, outputs, targets, epoch, indexs):
        self.ema[indexs] = self.w_ema * self.ema[indexs] + (1 - self.w_ema) * outputs.detach()
        ema_sm = F.softmax(self.ema[indexs] * self.temp, dim=-1).detach()
        alpha = torch.gather(ema_sm, 1, targets.view(-1, 1))
        weight = min(1., epoch/self.warmup)
        alpha = 0.5 * (weight * alpha + (1 - weight) * torch.ones(alpha.size()).cuda())
        alpha = 0.5 * torch.ones(alpha.size()).cuda()
        loss = self.loss_fn(outputs, targets, alpha.view(-1, 1))
        return loss
    
class SelfDistillLoss(nn.Module):
    """ Self-Distillation Loss for Final Classifer. """
    def __init__(self):
        super(SelfDistillLoss, self).__init__()
        self.loss_fn = NLLLoss()
        
    def forward(self, outputs, targets, indexs):
        alpha = torch.gather(F.softmax(outputs, dim=-1).detach(), 
                             1, targets.view(-1,1))
        loss = 0.3 * F.cross_entropy(outputs, targets)
        loss += 0.7 * self.loss_fn(outputs, targets, alpha.view(-1,1))
        return loss

class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, targets, alpha):
        targets = torch.zeros(outputs.size(0), outputs.size(1),
                              device=targets.device).scatter_(1, targets.view(-1,1), 1)
        all_one = torch.ones(outputs.size(), device=targets.device)
        
        targets = alpha * targets
        targets += (1 - alpha) / outputs.size(1) * all_one
        loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1))
        return loss
    
class LSLoss(nn.Module):
    """ Label Smoothing Loss. """
    def __init__(self, criterion, alpha=0.1, lam=1.0):
        super().__init__()
        self.criterion = criterion
        self.middle_loss_fn = NLLLoss()
        self.alpha = alpha
        self.lam = lam
        
    def forward(self, outputs, targets, targets_gt, indexs):
        loss = self.criterion(outputs[0], targets, indexs)
        for i in range(1, len(outputs)):
            alpha = self.alpha * torch.ones(outputs[0].size(0), device=targets.device)
            loss += self.lam * self.middle_loss_fn(outputs[i], targets, alpha.view(-1, 1))
        return loss
    
class ALSLoss(nn.Module):
    """ Adaptive Label Smoothing Loss. """
    def __init__(self, criterion, lam, num_examp, num_class):
        super().__init__()
        self.criterion = criterion
        self.middle_loss_fn = NLLLoss()
        self.lam = lam
        self.ema = torch.zeros((num_examp, num_class)).cuda() / num_class
        
    def forward(self, outputs, targets, epoch, indexs):
        
        # Compute Agreement
        if epoch > 20:
            _targets = torch.where(outputs[0].max(1)[1] == outputs[-1].max(1)[1],
                                   outputs[0].max(1)[1].detach(), targets)
        else:
            _targets = targets
        
        loss = self.criterion(outputs[0], targets, indexs)
        self.ema[indexs] = 0.7 * self.ema[indexs] + 0.3 * outputs[0].detach()
        ema_sm = F.softmax(self.ema[indexs] * 3.0, dim=-1).detach()
        alpha = torch.gather(ema_sm, 1, _targets.view(-1, 1))
        for i in range(1, len(outputs)):
            loss += self.lam * self.middle_loss_fn(outputs[i], _targets, alpha.view(-1, 1))
        return loss
    
class SDLoss(nn.Module):
    def __init__(self, criterion, lam, n_epochs=120, w_kd=0.7):
        super().__init__()
        self.criterion = criterion
        self.middle_loss_fn = NLLLoss()
        self.lam = lam
        
    def forward(self, outputs, targets, epoch, indexs):
        loss = self.criterion(outputs[0], targets, indexs)
        prob = torch.softmax(outputs[0], dim=-1).detach()
        alpha = torch.gather(prob, 1, targets.view(-1, 1))
        for i in range(1, len(outputs)):
            student_prob = F.log_softmax(outputs[i], dim=-1)
            loss += torch.mean(torch.sum(-prob * student_prob, dim=-1))
        return loss
    
class FDLoss(nn.Module):
    """ Feature Distillation Loss (MSE) """
    def __init__(self):
        super().__init__()
        
    def forward(self, feat1, feat2):
        loss = (feat1 - feat2)**2 * ((feat1 > 0) | (feat2 > 0)).float()
        return torch.abs(loss).sum()

class KDLoss(nn.Module):
    """ Vanilla Knowledge Distillation Loss """
    def __init__(self):
        super().__init__()
        self.temp = 2.0
        
    def forward(self, out1, out2):
        pred = F.softmax(out2/self.temp, dim=1)
        loss = -torch.mean(torch.sum(F.log_softmax(out1/self.temp, dim=1) * pred, dim=1))
        return loss

class BYOT(nn.Module):
    """BYOT Loss (be your own teacher)"""
    def __init__(self, kd=True, fd=True):
        super().__init__()
        self.fd_loss = FDLoss() if fd else None
        self.kd_loss = KDLoss() if kd else None
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.w1 = 0.1
        self.w2 = 1e-6
        
    def forward(self, outputs, feats, targets):
        
        if self.kd_loss is not None:
            loss = (1-self.w1) * self.ce_loss(outputs[0], targets)
            for i in range(1, len(outputs)):
                loss += self.w1 * self.kd_loss(outputs[i], outputs[0].detach())
        else:
            loss = self.ce_loss(outputs[0], targets)
            
        if self.fd_loss is not None:
            for i in range(1, len(feats)):
                loss += self.w2 * self.fd_loss(feats[i], feats[0].detach())
        return loss