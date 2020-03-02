#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from os.path import join

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo

import numpy as np
from  efficientnet_pytorch import EfficientNet

from torchsummary import summary

BatchNorm = nn.BatchNorm2d

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class EfficientNetSeg(nn.Module):
    def __init__(self, base_name, heads,
                 pretrained=True, head_conv=26):
        super(EfficientNetSeg, self).__init__()
        self.heads = heads

        print('efficientnet-b{}'.format(base_name))
        self.model = EfficientNet.from_pretrained('efficientnet-b{}'.format(base_name))
        self.model.to('cuda')
        summary(self.model, (3, 224, 224), device="cuda")


        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(1536, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.model.extract_features(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def get_pose_net(num_layers, heads, head_conv=256):
  model = EfficientNetSeg(num_layers, heads,
                 pretrained=True,
                 head_conv=head_conv)
  return model
