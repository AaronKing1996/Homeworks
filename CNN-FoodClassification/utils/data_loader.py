import torch
from torch.utils.data import Dataset, DataLoader


class ImageDataSet(Dataset):

    def __init__(self, x, y=None, transform=None):
        """
            实现抽象类DataSet的接口
        :param x: x值
        :param y: y值
        :param transform: 对x的变换，如数据增强
        """
        self.__x = x
        if y is not None:
            self.__y = torch.LongTensor(y)
        else:
            self.__y = None
        self.__transform = transform

    def __len__(self):
        """
            返回数据集的大小
        :return: 数据集大小
        """
        return len(self.__x)

    def __getitem__(self, index):
        """
            根据索引index返回数据
        :param index: 索引
        :return: (x,y)或x
        """
        x = self.__x[index]
        if self.__transform is not None:
            x = self.__transform(x)

        if self.__y is not None:
            y = self.__y[index]
            return x, y
        else:
            return x

    def get_loader(self, batch_size, shuffle=True):
        """
            返回DataLoader
        :param batch_size: batch大小
        :param shuffle: 是否shuffle
        :return: DataLoader
        """
        loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle)
        return loader
