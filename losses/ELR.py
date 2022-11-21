import torch.nn.functional as F
import torch
import torch.nn as nn

class ELRLoss(nn.Module):
    def __init__(self, num_examp, num_class=10, beta=0.7, lmbda=3.):
        super().__init__()
        self.num_classes = num_class
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.beta = beta
        self.lmbda = lmbda
        
    def forward(self, output, label, index):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss +  self.lmbda * elr_reg
        return final_loss