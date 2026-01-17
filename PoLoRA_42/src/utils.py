import random
import os, sys
from copy import deepcopy
import time

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F
import numpy as np
from prettytable import PrettyTable

""" Set random seeds """
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

""" Get learnable parameters """
def get_param(shape):
    param = Parameter(torch.Tensor(*shape)).double()
    xavier_normal_(param.data)
    return param