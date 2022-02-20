import torch
import torch.nn as nn
import torchvision.models  as models
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False

class projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(projector, self).__init__()
        self.layer1 = nn.Linear(in_dim, in_dim * 2)
        self.bn1 = nn.BatchNorm1d(in_dim * 2)
        self.layer2 = nn.Linear(in_dim * 2, in_dim)
        self.layer3 = nn.Linear(in_dim, out_dim)
        self.layer4 = nn.Linear(out_dim, out_dim * 2)
        self.bn2 = nn.BatchNorm1d(out_dim * 2)
        self.layer5 = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x):
        z1 = self.layer1(x)
        z1 = self.bn1(z1)
        z1 = self.layer2(z1)
        x = x + z1
        x = self.layer3(x)
        z2 = self.layer4(x)
        z2 = self.bn2(z2)
        z2 = self.layer5(z2)
        x = x + z2

        return x

        

class init_model(nn.Module):
    def __init__(self):
        super(init_model, self).__init__()
        self.bockbone = models.resnet50(pretrained=True)
        self.bockbone.fc = nn.Identity()
        freeze_parameters(self.bockbone, 50, train_fc=False)

        self.fc = MLP(2048, 1024)

    def forward(self, x):
        z = self.bockbone(x)
        # print(z.size())
        z = self.fc(z)
        z = F.normalize(z)

        return z

class continual_model(nn.Module):
    def __init__(self, pre_model, cur_model):
        super(continual_model, self).__init__()
        self.pre_model = pre_model
        self.cur_model = cur_model
        self.projector = projector(1024, 1024)

    def forward(self, x):
        pre_z = self.pre_model(x)
        pre_z = self.projector(pre_z)

        cur_z = self.cur_model(x)

        return pre_z, cur_z

    def get_cur_model(self):
        return self.cur_model

    def get_pre_model(self):
        return self.pre_model
























