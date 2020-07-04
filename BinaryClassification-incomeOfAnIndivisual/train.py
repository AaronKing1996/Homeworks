import numpy as np
import tools
import drawer


def _read_data(file_path):
    with open(file_path) as file:
        next(file)
        return np.array([line.strip('\n').split(',')[1:] for line in file], dtype=float)


def _logistic_regression(x, weight, bias):
    """
        逻辑回归
    :param x: input data
    :param weight:
    :param bias:
    :return:
    """
    # numpy.matmul 函数返回两个数组的矩阵乘积
    return tools.sigmoid(np.matmul(x, weight) + bias)


def _gradient(x, y_label, weight, bias):
    """
        交叉熵loss梯度下降
    :param x:
    :param y_label:
    :param weight:
    :param bias:
    :return:
    """
    # 预测值
    y_predict = _logistic_regression(x, weight, bias)
    y_error = y_label - y_predict
    # 按照行求和
    w_gradient = -np.sum(y_error.T * x.T, 1)
    b_gradient = -np.sum(y_error)
    return w_gradient, b_gradient


def _pre_processing(x_data, y_data, ratio):
    """
        数据预处理：标准化 + 拆分成验证集
    :param x_data:
    :param y_data:
    :param ratio: 验证集的比例
    :return:
    """
    print("Data pre-processing...")
    # 标准化
    x_normalized = tools.normalize(x_data)
    # 拆分成验证集
    return tools.split_to_validation(x_normalized, y_data, ratio)


def train(config):
    """
        训练
    :param config: 参数配置
    :return:
    """
    # training parameters
    max_iterations = int(config['TRAINING']['max_iterations'])
    training_batch = int(config['TRAINING']['training_batch'])
    learning_rate = float(config['TRAINING']['learning_rate'])
    validation_ratio = float(config['TRAINING']['validation_ratio'])

    x_data = _read_data(config['BASE']['x_train_path'])
    y_data = _read_data(config['BASE']['y_train_path'])
    # pre_processing
    x_train, y_train, x_validation, y_validation = _pre_processing(x_data, y_data, validation_ratio)

    # 初始化参数
    dim = x_train.shape[1]
    train_size = x_train.shape[0]
    validation_size = x_validation.shape[0]
    w = np.zeros([dim, 1])
    b = np.zeros([1, 1])
    adagrad = 0.0
    esp = 0.0000000001

    # 保存准确率和loss
    train_loss = []
    validation_loss = []
    train_acc = []
    validation_acc = []

    # training start
    for iterator in range(max_iterations):
        # 每一轮训练迭代前先随机打乱
        x_random, y_random = tools.shuffle(x_train, y_train)

        # 再从随机打乱的数据，分批次训练并更新w,b
        for batch in range(int(x_train.shape[0] / training_batch)):
            x_batch = x_random[training_batch * batch: training_batch * (batch + 1)]
            y_batch = y_random[training_batch * batch: training_batch * (batch + 1)]

            # 计算梯度
            w_gradient, b_gradient = _gradient(x_batch, y_batch, w, b)

            adagrad += b_gradient ** 2
            # 更新w,b
            w = w - learning_rate / np.sqrt(adagrad + esp) * w_gradient.reshape(-1, 1)
            b = b - learning_rate / np.sqrt(adagrad + esp) * b_gradient

        # 一轮结束，计算训练集和验证集的准确率和loss
        y_random_predict = _logistic_regression(x_random, w, b)
        train_acc.append(tools.accuracy(np.round(_logistic_regression(x_random, w, b)), y_random))
        train_loss.append(tools.cross_entropy_loss(y_random.T, y_random_predict)[0][0] / train_size)

        y_validation_predict = _logistic_regression(x_validation, w, b)
        validation_acc.append(tools.accuracy(np.round(_logistic_regression(x_validation, w, b)), y_validation))
        validation_loss.append(tools.cross_entropy_loss(y_validation.T, y_validation_predict)[0][0] / validation_size)

    # plot
    drawer.plot_two_dimensions("acc", [train_acc, validation_acc], ["train", "validation"], True)
    drawer.plot_two_dimensions("loss", [train_loss, validation_loss], ["train", "validation"], True)
