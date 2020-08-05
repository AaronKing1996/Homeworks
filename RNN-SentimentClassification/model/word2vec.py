import time
from pathlib import Path
from gensim.models import word2vec


def gensim_word2vec(sentences, path, rebuild=False):
    """
        生成词向量模型
    :param sentences: [list] 语料
    :param path: [string] 模型的保存位置
    :param rebuild: [bool] 是否重新生成
    :return:
    """
    if rebuild or Path(path).is_file() is False:
        print("开始训练词向量模型...")
        start = time.clock()
        model = word2vec.Word2Vec(sentences, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
        model.save(path)
        end = time.clock()
        print("训练完成，共耗时 %d 秒" % (end - start))
    else:
        print("词向量模型已存在！")
