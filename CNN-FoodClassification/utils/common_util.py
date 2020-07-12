import os
import numpy as np
import cv2 as cv
from tqdm import tqdm


def read_data(path, label, input_size):
    """
        读取数据，若label==True，则返回y值，否则只返回x值
    :param path: [string] 文件夹目录
    :param label: [boolean] 是否返回y值
    :param input_size: [int] 输入图片的大小 input_size*input_size
    :return:
    """
    print("读取数据%s..." % path)
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), input_size, input_size, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in zip(range(len(image_dir)), tqdm(image_dir)):
        img = cv.imread(os.path.join(path, file))
        x[i, :, :] = cv.resize(img, (input_size, input_size))
        if label:
            y[i] = int(file.split("_")[0])

    # 是否返回y值
    if label:
        return x, y
    else:
        return x
