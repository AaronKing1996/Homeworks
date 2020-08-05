import torch.nn as nn


def conv2d_layer(elem):
    return nn.Conv2d(
        in_channels=elem[0],
        out_channels=elem[1],
        kernel_size=elem[2],
        stride=elem[3],
        padding=elem[4]
    )


def linear_layer(elem):
    return nn.Linear(in_features=elem[0], out_features=elem[1])


def max_pool_2d(elem):
    return nn.MaxPool2d(kernel_size=elem[0], stride=elem[1], padding=elem[2])


def dropout(elem):
    return nn.Dropout(elem[0])


def relu(elem):
    return nn.ReLU(inplace=elem[0])


layer_switch = {
    'Conv2d': conv2d_layer,
    'Linear': linear_layer,
    'MaxPool2d': max_pool_2d,
    'Dropout': dropout,
    'ReLU': relu
}


def make_layers(net_structure):
    # 保存网络层
    layers = []
    for elem in net_structure:
        layers.append(layer_switch[elem[0]](elem[1:]))
    return nn.Sequential(*layers)
