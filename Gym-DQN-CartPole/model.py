#!/usr/bin/env python3
# UTF-8
import torch
import torch.nn as nn
import math
import numpy as np


def initialize_weights(modules):
    """
        初始化权重
    """
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            # kernel的体积
            kernel_volume = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            for i in range(layer.out_channels):
                # normal(mean, std)
                layer.weight.data[i].normal_(0, math.sqrt(2. / kernel_volume))
            if layer.bias is not None:
                layer.bias.data.zero_()
        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            layer.weight.data.normal_(0, 0.01)
            layer.bias.data.zero_()


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        initialize_weights(self.modules())

    def forward(self, x):
        return self.nn(x)


class DQNModel(nn.Module):
    def __init__(self, input_shape, output_size):
        super(DQNModel, self).__init__()
        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[-1], 168, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(168, 324, kernel_size=4, stride=2),
            nn.ReLU()
        )
        conv_out_size = self.__get_conv_out_size(input_shape)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        # 初始化权重
        initialize_weights(self.modules())

    def forward(self, x):
        # x.size()[0]为batch维度，size()[1:3]为一个batch输出
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def __get_conv_out_size(self, input_shape):
        out = self.conv(torch.zeros(1, input_shape[-1], *input_shape[:-1]))
        return int(np.prod(out.size()))
