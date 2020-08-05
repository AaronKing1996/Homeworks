import torchvision
import torch
import torchvision.transforms as tfs


class DataSet(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        # data是图片，target是文件名（类别），如第0个文件夹（即第0类）
        data, target = self.dt[index]
        return data, target

    def __len__(self):
        return len(self.dt)


def get_aug_dataloader(image_dir, is_validation=False,
                       batch_size=256, image_size=256, crop_size=224,
                       mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                       num_workers=8,
                       augs=1, shuffle=True):

    print(image_dir)
    if image_dir is None:
        return None

    print("image size: ", image_size, "crop size: ", crop_size)
    normalize = tfs.Normalize(mean=mean, std=std)
    if augs == 0:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.CenterCrop(crop_size),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 1:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.CenterCrop(crop_size),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 2:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.RandomResizedCrop(crop_size),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 3:
        _transforms = tfs.Compose([
                                    tfs.RandomResizedCrop(crop_size),
                                    tfs.RandomGrayscale(p=0.2),
                                    tfs.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])

    if is_validation:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/val', _transforms))
    else:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/train', _transforms))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return loader
