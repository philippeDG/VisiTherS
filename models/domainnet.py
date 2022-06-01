"""
proposed model with joint training of correlation and concatenation branches, with different feature extractions.

author: David-Alexandre Beaupre
date: 2020-04-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hrnet.seg_hrnet import get_seg_model
from models.features import Features
from models.classifier import Classifier


class DomainNet(nn.Module):
    def __init__(self, config):
        """
        represents the architecture of the proposed model.
        :param num_channels: number of channels of the input image.
        """
        super(DomainNet, self).__init__()
        # self.rgb_features = Features(num_channels=3)
        # self.lwir_features = Features(num_channels=3)
        self.rgb_features = get_seg_model(config)
        self.lwir_features = get_seg_model(config)
        self.correlation_cls = Classifier(num_channels=331776)#256
        self.concat_cls = Classifier(num_channels=663552) #512

    def forward(self, rgb: torch.Tensor, lwir: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        forward pass implementation of both correlation and concatenation branches.
        :param rgb: rgb patch tensor.
        :param lwir: lwir patch tensor.
        :return: 2 elements probability tensors (rgb and lwir being the same or not).
        """

        rgb,rgb_stage1 = self.rgb_features(rgb)
        lwir,lwir_stage1 = self.lwir_features(lwir)

        correlation = torch.matmul(rgb, lwir)
        concatenation = torch.cat((F.relu(rgb), F.relu(lwir)), dim=1)

        correlation = correlation.view(correlation.size(0), -1) # [64, 1539]
        concatenation = concatenation.view(concatenation.size(0), -1)

        correlation = self.correlation_cls(correlation)
        concatenation = self.concat_cls(concatenation)


        correlation_stage1 = torch.matmul(rgb_stage1, lwir_stage1)
        concatenation_stage1 = torch.cat((F.relu(rgb_stage1), F.relu(lwir_stage1)), dim=1)

        correlation_stage1 = correlation_stage1.view(correlation_stage1.size(0), -1) # [64, 1539]
        concatenation_stage1 = concatenation_stage1.view(concatenation_stage1.size(0), -1)

        correlation_stage1 = self.correlation_cls(correlation_stage1)
        concatenation_stage1 = self.concat_cls(concatenation_stage1)

        return correlation, concatenation, correlation_stage1, concatenation_stage1
