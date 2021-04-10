import importlib
from caflow.models.modules.blocks.AffineCouplingLayer import AffineCouplingOneSided
from caflow.models.modules.blocks.MLCouplingLayer import MLCouplingLayer

def coupling_layer(name):
    if name=='Affine':
        return AffineCouplingOneSided
    elif name=='MixLog':
        return MLCouplingLayer
    else:
        raise NotImplementedError
