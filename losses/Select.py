import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class CTSelect(object):
    def __init__(self, forget_rate, num_gradual, n_epoch):
        self.rate_schedule = self.generate_forget_rates(forget_rate, num_gradual, n_epoch)
        
    def __call__(self, outputs_1, outputs_2, targets, epoch):
        
        if isinstance(targets, tuple):
            targets1, targets2 = targets
        else:
            targets1, targets2 = targets, targets
        
        # model 1
        loss_1 = F.cross_entropy(outputs_1, targets1, reduction='none')
        ind_1_sorted = torch.argsort(loss_1.data).cuda()
        
        # model 2
        loss_2 = F.cross_entropy(outputs_2, targets2, reduction='none')
        ind_2_sorted = torch.argsort(loss_2.data).cuda()
        
        # sample small loss instances
        remember_rate = 1 - self.rate_schedule[epoch]
        num_remember =  int(remember_rate * len(outputs_1))
        
        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        return ind_1_update.detach(), ind_2_update.detach()
    
    def generate_forget_rates(self, forget_rate, num_gradual, n_epoch):
        rate_schedule = np.ones(n_epoch) * forget_rate
        rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)
        return rate_schedule
    
class DCSelect(object):
    def __init__(self, warmup_step=12000):
        self.warmup_step = warmup_step
    
    def __call__(self, outputs_1, outputs_2, targets, step):
        _, pred1 = torch.max(outputs_1.data, 1)
        _, pred2 = torch.max(outputs_2.data, 1)
        
        pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()
        logical_disagree_id = np.zeros(targets.size(), dtype=bool)
        disagree_id = []
        for idx, p1 in enumerate(pred1):
            if p1 != pred2[idx]:
                disagree_id.append(idx)
                logical_disagree_id[idx] = True
                
        disagree_id = np.array(disagree_id).astype(np.int64)
        int_logical_disagree_id = logical_disagree_id.astype(np.int64)
        _update_step = np.logical_or(int_logical_disagree_id, step<self.warmup_step).astype(np.float32)
        update_step = Variable(torch.from_numpy(_update_step).bool()).cuda()
        if len(disagree_id) > 0:
            return torch.from_numpy(disagree_id), torch.from_numpy(disagree_id)
        else:
            return update_step, update_step
        
class CTPSelect(object):
    def __init__(self, forget_rate, num_gradual, n_epoch, warmup_step=12000):
        self.rate_schedule = self.generate_forget_rates(forget_rate, num_gradual, n_epoch)
        self.warmup_step = warmup_step
        
    def __call__(self, outputs_1, outputs_2, targets, epoch, step):
        
        # DC Select
        _, pred1 = torch.max(outputs_1.data, 1)
        _, pred2 = torch.max(outputs_2.data, 1)
        
        pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()
        logical_disagree_id = np.zeros(targets.size(), dtype=bool)
        disagree_id = []
        for idx, p1 in enumerate(pred1):
            if p1 != pred2[idx]:
                disagree_id.append(idx)
                logical_disagree_id[idx] = True
                
        disagree_id = np.array(disagree_id).astype(np.int64)
        int_logical_disagree_id = logical_disagree_id.astype(np.int64)
        _update_step = np.logical_or(int_logical_disagree_id, step<self.warmup_step).astype(np.float32)
        update_step = Variable(torch.from_numpy(_update_step).bool()).cuda()
        
        if len(disagree_id) > 0:
            disagree_id = torch.from_numpy(disagree_id).cuda()
            outputs_1, outputs_2 = outputs_1[disagree_id], outputs_2[disagree_id]
            targets = targets[disagree_id]
        else:
            disagree_id = torch.arange(targets.size(0)).cuda()
            
        # model 1
        loss_1 = F.cross_entropy(outputs_1, targets, reduction='none')
        ind_1_sorted = torch.argsort(loss_1.data).cuda()
        
        # model 2
        loss_2 = F.cross_entropy(outputs_2, targets, reduction='none')
        ind_2_sorted = torch.argsort(loss_2.data).cuda()
        
        # sample small loss instances
        remember_rate = 1 - self.rate_schedule[epoch]
        num_remember =  int(remember_rate * len(outputs_1))
        
        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        return ind_1_update, ind_2_update
    
    def generate_forget_rates(self, forget_rate, num_gradual, n_epoch):
        rate_schedule = np.ones(n_epoch) * forget_rate
        rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)
        return rate_schedule