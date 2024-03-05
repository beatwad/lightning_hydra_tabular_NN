import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from argparse import Namespace

# define NN
class FCNet(nn.Module):
    """ Pytorch model
        Parameters
        ----------
        data_config
            LightningDataModule configuration parameters.
        fc_dims
            List of layers dimensions.
        """
    def __init__(self, data_config: dict, fc_dims: list) -> None:
        super().__init__()
        self.data_config = data_config
        self.input_width = int(np.prod(self.data_config["input_shape"][1:]))
        num_classes = len(self.data_config["mapping"])
        
        self.fc_dims = fc_dims
        num_layers = len(self.fc_dims)
        self.fc_dims = [self.input_width] + self.fc_dims

        self.layers = list()

        for i in range(num_layers):
            layer = nn.Sequential(OrderedDict([
                (f'fc{i+1}', torch.nn.Linear(self.fc_dims[i], self.fc_dims[i+1])),
                ('relu', nn.ReLU()),
                (f'bn{i+1}', torch.nn.BatchNorm1d(self.fc_dims[i+1]))]))
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)
        self.out = torch.nn.Linear(self.fc_dims[-1] + self.fc_dims[0], num_classes)

    def forward(self, x):  # overwrite this to use your nn.Modules from above
        x = torch.flatten(x, 1)
        identity = x
        for layer in self.layers:
            x = layer(x)
        x = torch.cat((x, identity), dim=1)
        return self.out(x)
