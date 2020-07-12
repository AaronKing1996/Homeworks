import torchvision.transforms as transforms


# 训练时进行增强
TRAINING_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()   # 将图片转为tensor，并normalize到[0，1]
])

# 测试时不进行增强
TESTING_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

