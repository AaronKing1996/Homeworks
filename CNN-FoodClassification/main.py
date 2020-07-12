import configparser
import torch.nn as nn
import train
from config.constant import *
from utils import common_util
from utils.data_loader import *
from model.cnn128 import CNN128


if __name__ == "__main__":
    # 参数配置
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    # 读取参数
    training_path = config['BASE']['training_path']
    validation_path = config['BASE']['validation_path']
    input_size = int(config['TRAINING']['input_size'])
    batch_size = int(config['TRAINING']['batch_size'])
    max_epoch = int(config['TRAINING']['max_epoch'])
    learning_rate = float(config['TRAINING']['learning_rate'])

    # 读取图片数据，并存放到numpy array中
    training_x, training_y = common_util.read_data(training_path, True, input_size)
    validation_x, validation_y = common_util.read_data(validation_path, True, input_size)

    # 加载图片数据
    training_loader = ImageDataSet(training_x, training_y, TRAINING_TRANSFORM).get_loader(batch_size)
    validation_loader = ImageDataSet(validation_x, validation_y, TESTING_TRANSFORM).get_loader(batch_size, False)

    # 训练模型
    model = CNN128().cuda()
    loss = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train = train.Train(model, loss, optimizer)
    train.training(training_loader, validation_loader, max_epoch)
    train.plot()
    pass
