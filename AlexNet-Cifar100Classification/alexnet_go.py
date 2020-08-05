import torch
from tensorboardX import SummaryWriter
from alexnet_model import AlexNet
import dataset
import files

"""
+image_folder
    +train
        +class_1
            image_1.jpg
            ...
         +class_2 
            image_1.jpg
            ...
    +val
        +class_1
            image_1.jpg
            ...
"""
image_folder = 'D:\Data\cifar-100-python\cifar-100-python'
checkpoint_dir = "./checkpoints"
weight_decay = 1e-5
momentum = 0.9
lr = 0.05
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 可视化
    writer = SummaryWriter(comment='_alexnet_go_Adam_lr={}_momentum={}_epochs={}'.format(lr, momentum, epochs))

    # 模型
    model = AlexNet(num_classes=100)
    model.to('cuda:0')
    writer.add_graph(model, torch.randn([256, 3, 224, 224]).cuda())

    # 损失
    loss_func = torch.nn.CrossEntropyLoss()

    # 优化器 SGD(loss func, grad func...)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                weight_decay=weight_decay,
                                momentum=momentum,
                                lr=lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.99))

    # 学习率更新
    lr_schedule = lambda epoch: ((epoch < epochs * 0.8) * lr * 0.95
                                 + (epoch >= epochs * 0.8) * lr * 0.8)

    # 数据集
    train_loader = dataset.get_aug_dataloader(image_folder, batch_size=128)

    for epoch in range(epochs):

        # model.train() ：启用BatchNormalization和Dropout, model.eval() ：不启用BatchNormalization和Dropout
        model.train()
        # 更新学习率
        lr = lr_schedule(epoch)
        print(f"Starting epoch {epoch}, lr = {lr} ...", flush=True)

        # optimizer.param_groups[0]：长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        for iter, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            # 前向传播
            pre_label = model(data)
            loss = loss_func(pre_label, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % 25 == 0:
                print('epoch:{}, loss:{:.4f}'.format(epoch, loss.item()))
                writer.add_scalar('loss', loss.item(), iter + epoch * len(train_loader))
                writer.add_scalar('lr', lr,  iter + epoch * len(train_loader))

        # 保存checkpoints
        files.save_checkpoint_all(checkpoint_dir, model, 'alexnet',
                                  optimizer, pre_label, epoch, lowest=False)

        _, predicted = torch.max(pre_label.data, 1)
        total = label.size(0)
        correct = (predicted == label).sum()
        print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct // total)))
        writer.add_scalar('correct', (100 * correct // total), (epoch + 1) * len(train_loader))

    writer.close()
