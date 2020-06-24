import numpy as np
import pandas as pd
import time


# read csv
def read_csv(file_path, skip_rows, skip_cols, item_count):
    """
    Args:
        file_path: CSV file path
        skip_rows: which row to read from
        skip_cols: which column to read from
        item_count: the count of items

    Returns:
        data matrix [row = itemCount]
    """

    # big5为繁体字编码
    with open(file_path, 'r', encoding='big5') as csv_file:
        # numpy 有缺值就无法读取
        # full_data = np.loadtxt(csv_file, delimiter=",", skiprows=skip_rows)
        full_data = pd.read_csv(csv_file, header=None)

        # 数据标签
        items = full_data.iloc[skip_rows:skip_rows + item_count, skip_cols]
        # 数据
        full_data[full_data == 'NR'] = 0
        disorder_data = full_data.iloc[skip_rows:, skip_cols:].to_numpy()

        # 将数据重新排序成行数为item_count的矩阵
        block_count = int(np.shape(disorder_data)[0] / item_count)
        block_cols = int(np.shape(disorder_data)[1])
        order_data = np.empty([item_count, block_count * block_cols], dtype=float)
        for num in range(block_count):
            block = disorder_data[num * item_count:(num + 1) * item_count, :]
            order_data[:, num * block.shape[1]:(num + 1) * block.shape[1]] = block

        return items, order_data


def pre_processing(data):
    """

    Args:
        data: processing data

    Returns:
        record : label
    """
    # 一个月有20天*24小时=480条观测数据，每条观测数据包含18项，每9小时为一次训练数据，共有480-9=471个训练数据
    # 按9小时的观测数据（共9*18项）reshape成行向量，以此组成x的行向量，第10小时的PM2.5为对应的label
    print("data pre processing...")
    # 将数据分为k份
    months_data = np.hsplit(data, 12)

    # 返回的数据和标签
    x = np.empty([12 * 471, 18 * 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)

    for month in range(12):
        month_data = months_data[month]
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                x[month * 471 + day * 24 + hour, :] = month_data[:, day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)
                y[month * 471 + day * 24 + hour, 0] = month_data[9, day * 24 + hour + 9]

    # normalize
    print("normalize...")
    # 压缩行，求列的平均值
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    for i in range(len(x)):
        for j in range(len(x[0])):
            if std_x[j] != 0:
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

    return x, y


def k_fold_cross_validation(data, k, learning_rate, output_file):
    """
    Args:
        learning_rate: learning rate
        data: order training data
        k: number of folds
        output_file: output file path
    Returns:

    """
    print("start cross validation...")
    start_time = time.clock()
    total_iter = 0
    total_loss = 0

    # 将训练集分成k份
    pre_processing_x, pre_processing_y = pre_processing(data)
    k_fold_x = np.vsplit(pre_processing_x, k)  # 12x471x(18*9)
    k_fold_y = np.vsplit(pre_processing_y, k)  # 12x471x1

    # 开始训练
    print("training...")

    for i in range(k):
        # 第一个fold为验证集
        validation_x = k_fold_x[0]
        validation_y = k_fold_y[0]
        del (k_fold_x[0])
        del (k_fold_y[0])
        training_x = k_fold_x
        training_y = k_fold_y

        # 初始化权重w，adagrad，迭代次数
        dim = 18 * 9 + 1
        w = np.zeros([dim, 1])
        adagrad = np.zeros([dim, 1])
        iter_count = 1000

        # eps避免adagrad的分母为零
        eps = 0.0000000001

        for it in range(iter_count):
            for count in range(len(training_x)):
                x = training_x[count]
                # 多了一个bias，数据集需要多一列1
                x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1).astype(float)
                y = training_y[count]
                # loss
                loss = np.sqrt(1 / (len(x)) * np.sum(np.power(np.dot(x, w) - y, 2)))
                # gradient = loss对w求偏导
                gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)
                # adagrad = gradient的平方的和
                adagrad += gradient ** 2
                w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
                # 评估
                total_iter += 1

        validate_count = 0
        validate_loss = 0
        # 验证
        validation_x_temp = np.concatenate((np.ones([validation_x.shape[0], 1]), validation_x), axis=1).astype(float)
        for (v_x, v_y) in zip(validation_x_temp, validation_y):
            predict_y = np.dot(v_x, w)
            if predict_y < 0:
                predict_y = 0
            predict_loss = np.power(predict_y - v_y, 2)
            validate_loss += predict_loss
            validate_count += 1

        # 总loss
        total_loss += abs(np.sqrt(validate_loss/validate_count))
        print("training round No.{" + str(i + 1) + "} validation loss = " + str(np.sqrt(validate_loss/validate_count)))

        # 将验证集push回去
        k_fold_x.append(validation_x)
        k_fold_y.append(validation_y)

    end_time = time.clock()
    print("learning rate = %d, time cost = %s seconds, iterate count = %d, average loss = %f" % (
        learning_rate, end_time - start_time, total_iter, total_loss / k))
    np.save( output_file + 'weight.npy', w)
