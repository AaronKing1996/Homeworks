import torch
import numpy as np
from tqdm import tqdm
from utils.drawer import *


class Train:
    def __init__(self, model, loss, optimizer):
        """
            初始化训练模型、loss、优化器
        :param model: 训练模型
        :param loss:
        :param optimizer: Adam、SGDM
        """
        self.__model = model
        self.__loss = loss
        self.__optimizer = optimizer
        # 保存准确率和loss
        self.__training_acc = []
        self.__training_loss = []
        self.__validation_acc = []
        self.__validation_loss = []

    def training(self, training_loader, validation_loader, max_epoch):
        """
            训练
        :param training_loader: [DataLoader] 训练数据集
        :param validation_loader:  [DataLoader] 验证数据集
        :param max_epoch:  [int] 最大迭代次数
        :return:
        """
        torch.cuda.empty_cache()

        for _ in tqdm(range(max_epoch), desc='训练中'):
            training_acc = 0.0
            training_loss = 0.0
            validation_acc = 0.0
            validation_loss = 0.0

            # 训练
            self.__model.train()
            for i, data in enumerate(training_loader):
                self.__optimizer.zero_grad()    # 将model的gradient清零
                training_predict = self.__model(data[0].cuda())    # 调用model的forward函数，得到预测概率分布
                batch_loss = self.__loss(training_predict, data[1].cuda())  # 计算loss
                batch_loss.backward()   # 利用back propagation计算每个参数的gradient
                self.__optimizer.step()     # 用gradient更新参数

                training_acc += np.sum(np.argmax(training_predict.cpu().data.numpy(), axis=1) == data[1].numpy())
                training_loss += batch_loss.item()

            # 验证
            self.__model.eval()
            with torch.no_grad():
                for i, data in enumerate(validation_loader):
                    validation_predict = self.__model(data[0].cuda())
                    batch_loss = self.__loss(validation_predict, data[1].cuda())

                    validation_acc += np.sum(np.argmax(validation_predict.cpu().data.numpy(), axis=1) == data[1].numpy())
                    validation_loss += batch_loss.item()

            self.__training_acc.append(training_acc / training_loader.dataset.__len__())
            self.__training_loss.append(training_loss / training_loader.dataset.__len__())
            self.__validation_acc.append(validation_acc / validation_loader.dataset.__len__())
            self.__validation_loss.append(validation_loss / validation_loader.dataset.__len__())
            pass

    def plot(self):
        # plot
        plot_two_dimensions("acc", [self.__training_acc, self.__validation_acc], ["train", "validation"], True)
        plot_two_dimensions("loss", [self.__training_loss, self.__validation_loss], ["train", "validation"], True)
