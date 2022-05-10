import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import torchvision.models as models
import torch.nn.functional as F
from PIL import ImageFilter
import random
from torch.utils import data
import glob
import os
from collections import OrderedDict

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

        self.sutu_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        # x_3 = self.sutu_transform(x)
        return x_1, x_2

class Transform_dog_vs_cat:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop([224, 224], scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.sutu_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        x_3 = self.sutu_transform(x)
        return x_1, x_2, x_3

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet152(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, train_fc=False)

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_loaders(dataset, label_class, batch_size):
    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10
        transform = transform_color
        coarse = {}
        trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
        testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        trainset_1 = ds(root='data', train=True, download=True, transform=Transform(), **coarse)
        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        trainset_1.data = trainset_1.data[idx]
        trainset_1.targets = [trainset_1.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0,
                                                  drop_last=False)
        return train_loader, test_loader, torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                                                      shuffle=True, num_workers=0, drop_last=False)
    else:
        print('Unsupported Dataset')
        exit()

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

def get_cifar100_loaders(dataset, label_class, batch_size):
    if dataset == "cifar100":
        ds = torchvision.datasets.CIFAR100
        transform = transform_color
        coarse = {}
        trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
        testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        trainset_1 = ds(root='data', train=True, download=True, transform=Transform(), **coarse)
        # print(len(trainset.targets))
        trainset.targets = sparse2coarse(trainset.targets)
        testset.targets = sparse2coarse(testset.targets)
        trainset_1.targets = sparse2coarse(trainset_1.targets)
        # print(len(trainset.targets))
        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        trainset_1.data = trainset_1.data[idx]
        trainset_1.targets = [trainset_1.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0,
                                                  drop_last=False)
        return train_loader, test_loader, torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                                                      shuffle=True, num_workers=0, drop_last=False)
    else:
        print('Unsupported Dataset')
        exit()




from torchvision.datasets import ImageFolder

def get_dog_vs_cat_loader(dataset, label_class, batch_size):
    if dataset == "dog_vs_cat":
        transform = transform_color

        if label_class == 'cat':
            trainset = ImageFolder(os.path.join(r'F:\old\e\anomaly-detection-dog-vs-cat\train_cat'),
                                   transform)
            testset = ImageFolder(os.path.join(r'F:\old\e\anomaly-detection-dog-vs-cat\test_cat_normaly'), transform)
            trainset_1 = ImageFolder(os.path.join(r'F:\old\e\anomaly-detection-dog-vs-cat\train_cat'),
                                     Transform_dog_vs_cat())
            # testset2 = ImageFolder(os.path.join(r'F:\old\e\anomaly-detection-dog-vs-cat\test_cat_normaly', '1'), transform)
            # testset = testset2 + testset1

        else:
            trainset = ImageFolder(os.path.join(r'F:\old\e\anomaly-detection-dog-vs-cat\train_dog'),
                                   transform)
            testset = ImageFolder(os.path.join(r'F:\old\e\anomaly-detection-dog-vs-cat\test_dog_normaly'),
                                  transform)
            trainset_1 = ImageFolder(os.path.join(r'F:\old\e\anomaly-detection-dog-vs-cat\train_dog'), Transform_dog_vs_cat())

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                                  drop_last=False)
        return train_loader, test_loader, torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                                                      shuffle=True, num_workers=2, drop_last=False)

    else:
        print('Unsupported Dataset')
        exit()


# def get_video_loader():






# if __name__ == '__main__':
#     loader1, loader2, loader3 = get_dog_vs_cat_loader('dog_vs_cat', 'dog', 32)
#     # loader1, loader2, loader3 = get_cifar100_loaders('cifar100', 0, 32)
#     print(len(loader1))
#     print(len(loader2))
#
#     for i, image in enumerate(loader3):
#         # print(image[0])
#         print(image[0][0].shape)
#         print(image[0][1].shape)
#         print(image[0][2].shape)
#         # print(image[1].size())
#         # print(image[2].size())
#         if i == 0:
#             break