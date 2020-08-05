# modify from DeepCluster repo: https://github.com/facebookresearch/deepcluster
import math
import torch.nn as nn
from model_util import make_layers


# (number of filters, kernel size, stride, pad)
# input = [3, 224, 224]
NET_STRUCTURE = {
    'features': [
        ('Conv2d', 3, 96, 11, 4, 2),        # (224 - 11 + 2 * 2) / 4 + 1 = 55   =>  [96, 55, 55]
        ('ReLU', True),                     # True: 进行覆盖计算
        ('MaxPool2d', 3, 2, 0),           # (55 - 3 + 2 * 0) / 2  + 1 = 27    =>  [96, 27, 27]
        ('Conv2d', 96, 256, 5, 1, 2),       # (27 - 5 + 2 * 2) / 1 + 1 = 27     =>  [256, 27, 27]
        ('ReLU', True),
        ('MaxPool2d', 3, 2, 0),           # (27 - 3 + 2 * 0) / 2 + 1 = 13     =>  [256, 13, 13]
        ('Conv2d', 256, 384, 3, 1, 1),       # (13 - 3 + 2 * 1) / 1 + 1 = 13     =>  [384, 13, 13]
        ('ReLU', True),
        ('Conv2d', 384, 384, 3, 1, 1),       # (13 - 3 + 2 * 1) / 1 + 1 = 13     =>  [384, 13, 13]
        ('ReLU', True),
        ('Conv2d', 384, 256, 3, 1, 1),       # (13 - 3 + 2 * 1) / 1 + 1 = 13     =>  [256, 13, 13]
        ('ReLU', True),
        ('MaxPool2d', 3, 2, 0)            # (13 - 3 + 2 * 0) / 2 + 1 = 6      =>  [256, 6, 6]
    ],
    'classifier': [
        # 0.5：每个神经元有50%的可能性失活
        ('Dropout', 0.5),
        ('Linear', 256 * 6 * 6, 4096),
        ('ReLU', True),
        ('Dropout', 0.5),
        ('Linear', 4096, 4096),
        ('ReLU', True)
    ]
}


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        """
            1、初始化网络结构
            2、初始化权重
        @param num_classes: 分类数
        """
        super(AlexNet, self).__init__()

        # 特征提取层（5层卷积）
        self.features = make_layers(NET_STRUCTURE['features'])
        # 分类层（2层全连接）
        self.classifier = make_layers(NET_STRUCTURE['classifier'])
        # 输出层（1层全连接）:注意输入特征维数应与分类层的输出相同
        self.output = make_layers([('Linear', 4096, num_classes)])

        self._initialize_weights()

    def forward(self, x):
        # x : [256, 3, 224, 224]
        x = self.features(x)                        # x : [256, 256, 6, 6]
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)                      # x : [256, 4096]
        x = self.output(x)                          # x : [256, num_classes]
        return x

    def _initialize_weights(self):
        """
            初始化权重
        """
        for layer in self.modules():
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
