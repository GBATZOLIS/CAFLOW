import torch
import torch.nn as nn
import torch.nn.functional as F

def concat_elu(x):
    """Concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU."""
    return F.elu(torch.cat((x, -x), dim=1))