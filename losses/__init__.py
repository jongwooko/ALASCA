from .LNL import GCELoss, SCELoss, NFLandMAE, NCEandMAE, NFLandRCE, NCEandRCE
from .LNL import SCELoss, NFLandMAE, NCEandMAE, NFLandRCE, NCEandRCE
from .ELR import ELRLoss
from .Select import DCSelect, CTSelect, CTPSelect
from .distill import LabelSmoothingLoss, AdaptiveLabelSmoothingLoss, SelfDistillLoss, EMA_ALS
from .distill import LSLoss, ALSLoss, SDLoss, FDLoss, BYOT
from .ALASCA import ALASCA, clothing1M_ALASCA

import torch.nn as nn
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, i):
        return super().forward(x, y)