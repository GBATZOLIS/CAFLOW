import torch.nn as nn
import numpy as np

class InvertibleScaling(nn.Module):
    def __init__(self, scaling_range):
        self.range = scaling_range
    
    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z -= self.range[0]
            z /= self.range[1] - self.range[0]
            ldj += -1 * np.log(self.range[1] - self.range[0]) * np.prod(z.shape[1:])
            return z, ldj
        else:
            z *= self.range[1] - self.range[0]
            z += self.range[0]
            ldj += np.log(self.range[1] - self.range[0]) * np.prod(z.shape[1:])
            return z, ldj

