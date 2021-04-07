from caflow.models.modules.networks.SimpleConvNet import SimpleConvNet
from caflow.models.modules.networks.nnflowpp import nnflowpp

def parse_nn_by_name(name):
    if name=='nnflowpp':
        return nnflowpp
    elif name=='SimpleConvNet':
        return SimpleConvNet
