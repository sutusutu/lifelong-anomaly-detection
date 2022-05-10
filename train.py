import sys
import time
import torch.multiprocessing
import torch
import model.model as model
import contrastive.utils as utils
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import torch.nn.functional as F
from contrastive.loss import contrastive_loss

from log import Logger


def get_score(model, train_loader, test_loader):
    # model.eval()
    train_feature_space = []
    with torch.no_grad():
        for (image, _) in tqdm(train_loader, desc='Train set feature getting....... '):

            image = image.cuda()
            # print(image.size())
            _, feature = model(image)
            train_feature_space.append(feature)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.cuda()
            _, features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space





def init_train(init_model, train_loader, test_loader, train_aug_loader, lr, epochs, task):
    init_model.eval()
    auc, feature_space = get_score(init_model, train_loader, test_loader)
    print('task {} init model auc for init center: {}'.format(task, auc))
    optimizer = optim.SGD(init_model.parameters(), lr=lr, weight_decay=0.0005)
    # optimizer = optim.Adam(init_model.parameters(), lr=lr, weight_decay=0.0005)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    center = F.normalize(center, dim=-1)
    center = center.cuda()
    max_auc = auc
    max_epoch = 0
    torch.save(init_model.state_dict(), 'task_{}_best_acu.pkl'.format(task))

    for epoch in range(epochs):
        # init_model.train()
        print('init model epoch {}.....'.format(epoch))
        for ((image1, image2), _) in tqdm(train_aug_loader, desc='Train......'):
            image1 = image1.cuda()
            image2 = image2.cuda()

            optimizer.zero_grad()

            _, image1_feature = init_model(image1)
            _, image2_feature = init_model(image2)

            image1_feature = image1_feature - center
            image2_feature = image2_feature - center

            center_loss = ((image1_feature ** 2).sum(dim=1)).mean() + ((image2_feature ** 2).sum(dim=1)).mean()
            loss = contrastive_loss(image1_feature, image2_feature) + center_loss
            loss.backward()
            optimizer.step()
        auc, _ = get_score(init_model, train_loader, test_loader)
        if max_auc < auc:
            max_auc = auc
            max_epoch = epoch
            torch.save(init_model.state_dict(), 'task_{}_best_acu.pkl'.format(task))

        print('task {} init model auc for {} epoch =  {} ---> max auc in {} = {}'.format(task, epoch, auc, max_epoch, max_auc))

    return center, init_model



def one_task_continual_train(pre_center, cur_center ,pre_model, cur_model, task, lr, epochs, task_train_dataloader, task_train_dataloader1, task_test_dataloader):
    print('continual start train task {}......'.format(task))
    # cur_model.eval()
    pre_model.eval()
    auc, feature_space = get_score(pre_model, task_train_dataloader, task_test_dataloader)
    print('continual task {} init auc {}'.format(task, auc))
    #
    pre_center = pre_center.cuda()
    center = torch.FloatTensor(feature_space).mean(dim=0)
    center = F.normalize(center, dim=-1)
    center = center.cuda()
    # center = (pre_center + center) / 2
    # center = F.normalize(center, dim=-1)

    # center = pre_center + center
    # center = center / 2
    # center = center.cuda()
    # center = cur_center.cuda()
    # center = pre_center.cuda()


    continual_model = model.continual_model(pre_model, cur_model)
    continual_model = continual_model.cuda()

    optimizer = optim.SGD(continual_model.parameters(), lr=lr, weight_decay=0.0005)
    # optimizer = optim.Adam(continual_model.parameters(), lr=lr, weight_decay=0.0005)

    max_auc = 0
    max_epoch = 0
    torch.save(continual_model.get_cur_model().state_dict(), 'task_{}_best_acu.pkl'.format(task))
    for epoch in range(epochs):
        # continual_model.train()
        for ((image1, image2), _) in tqdm(task_train_dataloader1, desc='Train......'):
            image1 = image1.cuda()
            image2 = image2.cuda()

            pre_fea_1, pre_z_1, cur_fea_1, cure_z_1, p1 = continual_model(image1)
            pre_fea_2, pre_z_2, cur_fea_2, cure_z_2, p2 = continual_model(image2)

            optimizer.zero_grad()

            cure_z_1 = cure_z_1 - center
            cure_z_2 = cure_z_2 - center

            distill_loss = (F.mse_loss(p1, pre_z_1) + F.mse_loss(p2, pre_z_2)) / 2

            cur_shift_loss = contrastive_loss(cure_z_1, cure_z_2)
            center_loss = ((cure_z_1 ** 2).sum(dim=1)).mean() + ((cure_z_2 ** 2).sum(dim=1)).mean()


            sum_loss = 1 * distill_loss + (cur_shift_loss + center_loss) * 1

            sum_loss.backward()
            optimizer.step()

        auc, _ = get_score(continual_model.get_cur_model(), task_train_dataloader, task_test_dataloader)
        if epoch == epochs/2:
            torch.save(continual_model.get_cur_model().state_dict(), 'task_{}_mid_acu.pkl'.format(task))
            torch.save(continual_model.state_dict(), 'task_con_{}_mid.pkl'.format(task))
        if max_auc < auc:
            max_auc = auc
            max_epoch = epoch
            torch.save(continual_model.get_cur_model().state_dict(), 'task_{}_best_acu.pkl'.format(task))
            torch.save(continual_model.state_dict(), 'task_con_{}_best.pkl'.format(task))

        print('continual task {} model auc for {} epoch =  {} ---> max auc in {} = {}'.format(task, epoch, auc, max_epoch, max_auc))
        torch.save(continual_model.get_cur_model().state_dict(), 'task_{}_cur_acu.pkl'.format(task))
        # torch.save(continual_model.state_dict(), 'task_con_{}_cur_acu.pkl'.format(task))

        for i in range(task):
            if i >= task-1:
                pass
            else:
                test_continual('task_{}_cur_acu.pkl'.format(task), i)


    torch.save(continual_model.get_cur_model().state_dict(), 'task_{}_fin_acu.pkl'.format(task))
    # torch.save(continual_model.state_dict(), 'task_con_{}_fin.pkl'.format(task))
    return continual_model.get_cur_model(), center


def test_continual(path, i):
    test_continual_model = model.init_model()
    test_continual_model.load_state_dict(torch.load(path))
    test_continual_model = test_continual_model.cuda()
    with torch.no_grad():
        auc, feature_space = get_score(test_continual_model, train_dataloader_set[i], test_dataloader_set[i])
        print('for data-- {} -- continual learning testing for {} ....auc= {}...'.format(i, path,auc))


def continual_train(train_dataloader_set, train_aug_dataloader_set, test_dataloader_set, epochs, lr, all_task):
    init_model = model.init_model()
    init_model = init_model.cuda()
    pre_center, pre_model = init_train(init_model, train_dataloader_set[0], test_dataloader_set[0], train_aug_dataloader_set[0], lr[0], epochs[0], 1)

    for task in range(2, all_task+1):
        pre_model = model.init_model()
        print('load pre task_{}_best_acu.pkl'.format(task-1))
        pre_model.load_state_dict(torch.load('task_{}_best_acu.pkl'.format(task-1)))
        pre_model = pre_model.cuda()
        cur_model = None
        _, pre_center = one_task_continual_train(pre_center, None, pre_model, cur_model, task, lr[task-1], epochs[task-1], train_dataloader_set[task-1],
                                 train_aug_dataloader_set[task-1], test_dataloader_set[task-1])
        print('testing continual learning......')
        for i in range(task):
            if i >= task-1:
                pass
            else:
                test_continual('task_{}_best_acu.pkl'.format(task), i)
                test_continual('task_{}_mid_acu.pkl'.format(task), i)
                test_continual('task_{}_fin_acu.pkl'.format(task), i)







if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    train_dataloader_set = []
    train_aug_dataloader_set = []
    test_dataloader_set = []
    batch_size = 32

    sys.stdout = Logger('./exp_cifar10_cifar100.txt')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('starting process get dataloader.......')




    import dataloader.utils1 as cifar_utils
    train_path1_dataloader1,  test_path1_dataloader ,train_path1_dataloader2 = cifar_utils.get_loaders('cifar10', 0, 32)

    train_path2_dataloader1, test_path2_dataloader, train_path2_dataloader2 = cifar_utils.get_cifar100_loaders('cifar100', 0, 32)


    train_dataloader_set.append(train_path1_dataloader1)
    train_dataloader_set.append(train_path2_dataloader1)

    train_aug_dataloader_set.append(train_path1_dataloader2)
    train_aug_dataloader_set.append(train_path2_dataloader2)
    test_dataloader_set.append(test_path1_dataloader)
    test_dataloader_set.append(test_path2_dataloader)


    print('Ending process get dataloader.......')
    print('starting process training.......')
    print('lr=:{}-{}'.format(0.0002, 0.0002))
    print('task: bach, break')
    continual_train(train_dataloader_set, train_aug_dataloader_set, test_dataloader_set, epochs=[100, 100], lr=[0.0002, 0.0002],
                    all_task=2)




































