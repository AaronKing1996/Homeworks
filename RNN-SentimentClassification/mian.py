import configparser
from utils import common_util
from model import word2vec


if __name__ == '__main__':
    # 参数配置
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    training_label_path = config['BASE']['training_label_path']
    training_nolabel_path = config['BASE']['training_nolabel_path']
    testing_path = config['BASE']['testing_path']
    word2vec_model_path = config['BASE']['word2vec_model_path']

    # 读取数据
    x_training_labeled, y_training_labeled = common_util.read_training_data(training_label_path, True)
    x_training_no_labeled = common_util.read_training_data(training_nolabel_path, False)
    x_testing = common_util.read_testing_data(testing_path)

    # 词向量模型
    word2vec.gensim_word2vec(x_training_labeled + x_testing, word2vec_model_path)
