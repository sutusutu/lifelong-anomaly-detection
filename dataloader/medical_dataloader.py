import os
import torch
import torchvision.datasets as dset
import random
from PIL import ImageFilter
import torchvision.transforms as transforms


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

moco_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Transform:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


def get_train_dataset(path, batch_size):
    dataset1 = dset.ImageFolder(root=path, transform=transform_color)
    dataset2 = dset.ImageFolder(root=path, transform=Transform())
    train_dataset1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=2,
                                                 drop_last=False
                                                 )

    train_dataset2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=True, num_workers=2,
                                                 drop_last=False
                                                 )

    return train_dataset1, train_dataset2


def get_test_dataset(path, batch_size):
    dataset1 = dset.ImageFolder(root=path, transform=transform_color)
    test_dataset1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=2,
                                                drop_last=False)
    return test_dataset1
