
from torch import nn




def connector(connector_type='linear', **kwargs):
    print("Build connector:", connector_type)
    if connector_type == 'linear':
        return nn.ModuleList([nn.Linear(kwargs['input_dim'], kwargs['output_dim']) for i in range(kwargs['num_layers'])])
    else:
        raise NotImplemented
    





