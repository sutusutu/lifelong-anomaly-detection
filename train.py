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
import dataloader.medical_dataloader as med_data
from log import Logger

def get_score(model, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (image, _) in tqdm(train_loader, desc='Train set feature getting....... '):

            image = image.cuda()
            # print(image.size())
            feature = model(image)
            train_feature_space.append(feature)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.cuda()
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space



def init_train(init_model, train_loader, test_loader, train_aug_loader, lr, epochs):
    init_model.eval()
    auc, feature_space = get_score(init_model, train_loader, test_loader)
    print('init model auc for init center: {}'.format(auc))
    optimizer = optim.SGD(init_model.parameters(), lr=lr, weight_decay=0.0005)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    center = F.normalize(center, dim=-1)
    center = center.cuda()
    max_auc = auc
    max_epoch = 0

    for epoch in range(epochs):
        print('init model epoch {}.....'.format(epoch))
        for ((image1, image2), _) in tqdm(train_aug_loader, desc='Train......'):
            image1 = image1.cuda()
            image2 = image2.cuda()

            optimizer.zero_grad()

            image1_feature = init_model(image1)
            image2_feature = init_model(image2)

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
            torch.save(init_model.state_dict(), 'task_1_best_acu.pkl')

        print('init model auc for {} epoch =  {} ---> max auc in {} = {}'.format(epoch, auc, max_epoch, max_auc))

    return center

def frozen_pre_model(model):
    for layer in model.children():
        for p in layer.parameters():
            p.requires_grad = False

def one_task_continual_train(pre_center ,pre_model, cur_model, task, lr, epochs, task_train_dataloader, task_train_dataloader1, task_test_dataloader):
    print('start train task {}......'.format(task))
    frozen_pre_model(pre_model)
    cur_model.eval()
    auc, feature_space = get_score(cur_model, task_train_dataloader, task_test_dataloader)
    print('task {} init auc {}'.format(task, auc))
    center = torch.FloatTensor(feature_space).mean(dim=0)
    center = F.normalize(center, dim=-1)
    center = center.cuda()
    continual_model = model.continual_model(pre_model, cur_model)
    continual_model = continual_model.cuda()
    optimizer = optim.SGD(continual_model.parameters(), lr=lr, weight_decay=0.0005)
    max_auc = auc
    max_epoch = 0
    torch.save(continual_model.get_cur_model().state_dict(), 'task_{}_best_acu.pkl'.format(task))
    for epoch in range(epochs):
        for ((image1, image2), _) in tqdm(task_train_dataloader1, desc='Train......'):
            image1 = image1.cuda()
            image2 = image2.cuda()

            pre_image1, cur_image1 = continual_model(image1)
            pre_image2, cur_image2 = continual_model(image2)

            optimizer.zero_grad()
            pre_image1 = pre_image1 - pre_center
            pre_image2 = pre_image2 - pre_center


            cur_image1 = cur_image1 - center
            cur_image2 = cur_image2 - center

            loss1 = contrastive_loss(pre_image1.detach(), cur_image1)
            loss2 = contrastive_loss(pre_image2.detach(), cur_image2)

            loss3 = contrastive_loss(cur_image1, cur_image2)
            loss4 = ((cur_image1 ** 2).sum(dim=1)).mean() + ((cur_image2 ** 2).sum(dim=1)).mean()
            loss = (loss1 + loss2) * 0.5 + loss3 + loss4
            loss.backward()
            optimizer.step()

        auc, _ = get_score(cur_model, task_train_dataloader, task_test_dataloader)
        if max_auc < auc:
            max_auc = auc
            max_epoch = epoch
            torch.save(continual_model.get_cur_model().state_dict(), 'task_{}_best_acu.pkl'.format(task))

        print('task {} model auc for {} epoch =  {} ---> max auc in {} = {}'.format(task, epoch, auc, max_epoch, max_auc))

    return center


def continual_train(train_dataloader_set, train_aug_dataloader_set, test_dataloader_set, epochs, lr, all_task):
    init_model = model.init_model()
    init_model = init_model.cuda()
    pre_center = init_train(init_model, train_dataloader_set[0], test_dataloader_set[0], train_aug_dataloader_set[0], lr, epochs[0])

    for task in range(2, all_task+1):
        pre_model = model.init_model()
        pre_model.load_state_dict(torch.load('task_{}_best_acu.pkl'.format(task-1)))
        cur_model = model.init_model()
        pre_model = pre_model.cuda()
        cur_model = cur_model.cuda()
        pre_center = one_task_continual_train(pre_center, pre_model, cur_model, task, lr, epochs[task-1], train_dataloader_set[task-1],
                                 train_aug_dataloader_set[task-1], test_dataloader_set[task-1])
        print('testing continual learning......')
        for i in range(task):
            if i >= task-1:
                pass
            else:
                test_continual_model = model.init_model()
                test_continual_model.load_state_dict(torch.load('task_{}_best_acu.pkl'.format(task)))
                test_continual_model = test_continual_model.cuda()
                auc, feature_space = get_score(test_continual_model, train_dataloader_set[i], test_dataloader_set[i])
                print('for data-- {} -- continual learning testing....auc= {}...'.format(i, auc))




if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    train_dataloader_set = []
    train_aug_dataloader_set = []
    test_dataloader_set = []
    batch_size = 64
    sys.stdout = Logger('/root/autodl-tmp/fin/version_2/exp8.txt')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('starting process get dataloader.......')
    medical_path1_train = '/root/autodl-tmp/medical/BACH/train_test/train'
    medical_path2_train = '/root/autodl-tmp/medical/BreaKHis_v1/train_test/train'
    medical_path3_train = '/root/autodl-tmp/medical/Dataset_BUSI_with_GT/train_test/train'

    medical_path1_test = '/root/autodl-tmp/medical/BACH/train_test/test'
    medical_path2_test = '/root/autodl-tmp/medical/BreaKHis_v1/train_test/test'
    medical_path3_test = '/root/autodl-tmp/medical/Dataset_BUSI_with_GT/train_test/test'

    train_path1_dataloader1, train_path1_dataloader2 = med_data.get_train_dataset(medical_path1_train,
                                                                                  batch_size=batch_size)
    test_path1_dataloader = med_data.get_test_dataset(medical_path1_test, batch_size=batch_size)




    train_path2_dataloader1, train_path2_dataloader2 = med_data.get_train_dataset(medical_path2_train,
                                                                                  batch_size=batch_size)
    test_path2_dataloader = med_data.get_test_dataset(medical_path2_test, batch_size=batch_size)



    train_path3_dataloader1, train_path3_dataloader2 = med_data.get_train_dataset(medical_path3_train,
                                                                                  batch_size=batch_size)
    test_path3_dataloader = med_data.get_test_dataset(medical_path3_test, batch_size=batch_size)



    train_dataloader_set.append(train_path2_dataloader1)
    train_dataloader_set.append(train_path1_dataloader1)
    train_dataloader_set.append(train_path3_dataloader1)

    train_aug_dataloader_set.append(train_path2_dataloader2)
    train_aug_dataloader_set.append(train_path1_dataloader2)
    train_aug_dataloader_set.append(train_path3_dataloader2)

    test_dataloader_set.append(test_path2_dataloader)
    test_dataloader_set.append(test_path1_dataloader)
    test_dataloader_set.append(test_path3_dataloader)

    print('Ending process get dataloader.......')
    print('starting process training.......')
    print('lr=:{}'.format(0.00025))
    print('task: 2, 1, 3')
    continual_train(train_dataloader_set, train_aug_dataloader_set, test_dataloader_set, epochs=[200, 200, 300], lr=0.00025,
                    all_task=3)




































