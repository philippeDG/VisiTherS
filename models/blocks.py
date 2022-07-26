"""
gathering of blocks/group of layers used in stereohrnet.

author: David-Alexandre Beaupre
date: 2020-04-27

Modified by: Philippe Duplessis-Guindon
date: 2022-07-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearReLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        """
        represents the operations of a fully connected layer (require parameters) and ReLU (no parameters).
        :param in_dim: number of channels for the input.
        :param out_dim: number of channels for the output.
        :param bias: learn the linear bias or not.
        """
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass implementation (relu -> fc)
        :param x: input tensor.
        :return: tensor.
        """
        return F.relu(self.linear(x))

