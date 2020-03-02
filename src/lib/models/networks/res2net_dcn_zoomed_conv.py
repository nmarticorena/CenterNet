# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import torch
import torch.nn as nn
from .DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {
    'res2net50_26w_4s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net101_26w_4s-02a759a1.pth',
}

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 depthwise=False, bias=True, width_mult_list=[1.]):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)

    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input):
        assert self.ratio[0] in self.width_mult_list, str(self.ratio[0]) + " in? " + str(self.width_mult_list)
        self.in_channels = make_divisible(self.in_channels_max * self.ratio[0])
        assert self.ratio[1] in self.width_mult_list, str(self.ratio[1]) + " in? " + str(self.width_mult_list)
        self.out_channels = make_divisible(self.out_channels_max * self.ratio[1])
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return y

class conv3x3old(nn.Module):
    def  __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super(conv3x3old,self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    def forward(self,input):
        out = self.conv(input)
        return out

def conv3x3(in_planes, out_planes, stride=1,kernel_size=3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class zoomedConv3x3(nn.Module):
    def __init__(self, inplanes, planes, stride=1,kernel_size=3, padding=1, bias=False,dilation=1):
        super(zoomedConv3x3, self).__init__()
        self.stride=stride
        self.dilation = dilation
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)
        self.conv = nn.Conv2d(inplanes, planes,kernel_size=3,stride=1,groups=1,
                     padding=self.dilation,dilation=self.dilation, bias=bias)
        #self.conv= USConv2d(inplanes, planes,kernel_size=3,stride=stride,
        #            padding=self.dilation,dilation=self.dilation, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes,momentum=BN_MOMENTUM)


    def forward(self, x):
        out = torch.nn.functional.interpolate(x,size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear',align_corners=True)
        out = self.conv(out)
        out = self.bn1(out)
        if self.stride==1:
            out = torch.nn.functional.interpolate(out,size=(int(x.size(2)), int(x.size(3))), mode='bilinear',align_corners=True)
        return out

class zoomedConv3x32X(nn.Module):
    def __init__(self, inplanes, planes, stride=1,kernel_size=3, padding=1, bias=False,dilation=1):
        super(zoomedConv3x32X, self).__init__()
        self.stride=stride
        self.dilation = dilation
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)
        self.conv = nn.Conv2d(inplanes, planes,kernel_size=3,stride=1,groups=1,
                     padding=self.dilation,dilation=self.dilation, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes,momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, groups=1,
                              padding=self.dilation, dilation=self.dilation, bias=bias)
        self.bn2= nn.BatchNorm2d(planes,momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = torch.nn.functional.interpolate(x,size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear',align_corners=True)
        out = self.conv(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.stride == 1:
            out = torch.nn.functional.interpolate(out,size=(int(x.size(2)), int(x.size(3))), mode='bilinear',align_corners=True)
        out = self.relu(out)
        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,baseWidth=4,stype=None,scale=None):
        super(BasicBlock, self).__init__()
        #if stride==1:
        self.conv1 = zoomedConv3x3(inplanes, planes, stride)
        self.conv2 = zoomedConv3x3(planes, planes)
        #self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        """
        else:
            self.conv1 = conv3x3(inplanes,planes,stride)
            self.conv2 = conv3x3(planes,planes)
        """
        self.relu = nn.ReLU(inplace=True)

        #self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, scale=4, baseWidth=26,stype='normal'):
        super(Bottleneck, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale, momentum=BN_MOMENTUM)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []

        for i in range(self.nums):
            if stride==1:
                convs.append(zoomedConv3x3(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
                bns.append(None)
            else:
                convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
                bns.append(nn.BatchNorm2d(width, momentum=BN_MOMENTUM))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.width = width
        self.scale = scale
        self.stype=stype

    def forward(self, x):
        residual = x
        """ TODO Revisar
        out = nn.functional.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear')
        out = self.conv1(out)
        out = nn.functional.interpolate(out, size=(int(x.size(2)) * 2, int(x.size(3)) * 2), mode='bilinear')
        out = self.bn1(out)
        out = self.relu(out)
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype=='stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if self.bns[i]!=None:
                sp = self.relu(self.bns[i](sp))
            else:
                sp= self.relu(sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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


class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv,baseWidth=26,scale=4):
        self.baseWidth=baseWidth
        self.scale=scale


        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(64, head_conv,
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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, stype='stage', baseWidth=self.baseWidth,
                            scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes,
                     kernel_size=(3, 3), stride=1,
                     padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1, 
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #print (x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #print (self.heads)
        #print(x.shape)
        x = self.deconv_layers(x)
        #print(x.shape)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers,baseWidth=26,scale=4):
        print(num_layers)
        if 1:
            url = model_urls['res2net{}_{}w_{}s'.format(num_layers,baseWidth,scale)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv=256):
    print("Res2NetZoomed")
    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
    #model.init_weights(num_layers)
    return model
