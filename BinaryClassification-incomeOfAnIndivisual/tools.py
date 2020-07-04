import numpy as np


def shuffle(x, y):
    """
        数据随机洗牌
    :param x: vector
    :param y: vector
    :return:
    """
    randomize = np.arange(len(y))
    np.random.shuffle(randomize)
    return x[randomize], y[randomize]


def normalize(x):
    """
        将x标准化
    :param x: matrix
    :return: 标准化后的matrix
    """

    # 将行压缩，求每列的平均值和方差
    x_mean = np.mean(x, axis=0).reshape(1, -1)
    x_std = np.std(x, axis=0).reshape(1, -1)
    x_normalized = (x - x_mean) / (x_std + 1e-8)

    return x_normalized


def split_to_validation(x, y, ratio=0.25):
    """
        将训练集分成训练集和验证集
    :param x: 训练集x
    :param y: 训练集y
    :param ratio: 验证集的比例
    :return:
    """
    train_size = int(len(x) * (1 - ratio))
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:]


def sigmoid(x):
    """
        sigmoid函数
    :param x:
    :return:
    """
    # clip将数组中的元素限制在min, max之间
    return np.clip(1 / (1.0 + np.exp(-x)), 1e-8, 1-1e-8)


def cross_entropy_loss(x, y):
    """
        计算x与y之间的交叉熵
    :param x: bool vector
    :param y: float vector
    :return:
    """
    cross_entropy = - np.dot(x, np.log(y)) - np.dot(1-x, np.log(1-y))
    return cross_entropy


def accuracy(y_predict, y_label):
    """
        计算正确率
    :param y_predict: float vector
    :param y_label: float vector
    :return:
    """
    acc = 1 - np.mean(np.abs(y_predict - y_label))
    return acc
