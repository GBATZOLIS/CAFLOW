import torch
import torch.nn as nn
import torch.nn.functional as F

def concat_elu(x):
    """Concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU."""
    return F.elu(torch.cat((x, -x), dim=1))

def safe_log(x):
    return torch.log(x.clamp(min=1e-22))

def squeeze(x):
    ##-> provide the squeezing code for three dimensions as well

    """Trade spatial extent for channels. I.e., convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    if isinstance(x, list):
        new_tensor_list = []
        for tensor in x:
            b, c, h, w = tensor.size()
            tensor = tensor.view(b, c, h // 2, 2, w // 2, 2)
            tensor = tensor.permute(0, 1, 3, 5, 2, 4).contiguous()
            tensor = tensor.view(b, c * 2 * 2, h // 2, w // 2)
            new_tensor_list.append(tensor)
        return new_tensor_list
    else:
        b, c, h, w = x.size()
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)
        return x