"""Useful utils
"""
from .misc import *
from .logger import *
from .eval import *

# progress bar
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar

def set_seed(seed: int = 42):
    
    import random
    import numpy as np
    import os
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True