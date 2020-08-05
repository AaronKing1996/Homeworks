import torch
from tqdm import tqdm
from alexnet_model import AlexNet
import dataset
import files

image_folder = 'D:\Data\cifar-100-python\cifar-100-python'
checkpoint_dir = "./checkpoints"
weight_decay = 1e-5
momentum = 0.8
lr = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 模型
    model = AlexNet(num_classes=100)
    model.to('cuda:0')
    model.eval()

    # 优化器 SGD(loss func, grad func...)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                weight_decay=weight_decay,
                                momentum=momentum,
                                lr=lr)

    L, epoch = files.load_checkpoint_all(checkpoint_dir=checkpoint_dir, model=model, opt=optimizer)

    # 数据集
    test_loader = dataset.get_aug_dataloader(image_folder, batch_size=128, is_validation=True)

    total = 0
    correct = 0
    for data, label in tqdm(test_loader, desc='测试中...'):
        data = data.to(device)
        label = label.to(device)

        # 前向传播
        pre_label = model(data)

        # 误差
        _, predicted = torch.max(pre_label.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum()

    print('total = %d, correct = %d, 测试集准确率：%d%%' % (total, correct, 100 * correct // total))
