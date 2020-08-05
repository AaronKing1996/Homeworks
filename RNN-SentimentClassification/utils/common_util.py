def read_training_data(path, label):
    """
        读入训练集（包括有标签和无标签）
    :param path:  文件名
    :param label:   是否读入标签
    :return: x, y
    """
    with open(path, encoding='utf8') as file:
        content = file.readlines()
        # strip去掉首尾
        lines = [line.strip('\n').split(" ") for line in content]
        if label:
            y = [line[0] for line in lines]
            x = [line[2:] for line in lines]
            return x, y
        else:
            return lines


def read_testing_data(path):
    """
        读取测试集数据
    :param path:   测试集目录
    :return:    x值
    """
    with open(path, encoding='utf8') as file:
        content = file.readlines()
        # join将字符插入数组元素之间
        lines = ["".join(line.strip('\n').split(',')[1:]).split(' ') for line in content[1:]]
        return lines
