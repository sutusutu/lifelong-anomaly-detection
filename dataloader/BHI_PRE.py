import os
import shutil

def CreateDir(path):
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)
        print(path+' 目录创建成功')
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')

def get_all_path(path):
    file_path = os.listdir(path)
    file_path = file_path[:279]
    print(len(file_path))
    # file_path = file_path[:282]
    # path(len(file_path))
    # print(file_path)
    all_train_path = file_path[:200]
    print(len(all_train_path))
    all_test_path = file_path[200:]
    print(len(all_test_path))
    all_train_path_1 = []
    all_train_path_0 = []
    all_test_path_1 = []
    all_test_path_0 = []

    for sub_path in all_train_path:
        train_path_1 = path + '\\' + sub_path + '\\' + '1'
        train_path_0 = path + '\\' + sub_path + '\\' + '0'
        all_train_path_1.append(train_path_1)
        all_train_path_0.append(train_path_0)

    for sub_path in all_test_path:
        test_path_1 = path + '\\' + sub_path + '\\' + '1'
        test_path_0 = path + '\\' + sub_path + '\\' + '0'
        all_test_path_1.append(test_path_1)
        all_test_path_0.append(test_path_0)

    print(all_train_path_1)
    print(len(all_train_path_1))

    return all_train_path_1, all_train_path_0 , all_test_path_1, all_test_path_0

def Copy_File(in_path, out_path):
    for sub_in_path in in_path:
        file_path = os.listdir(sub_in_path)
        for file in file_path:
            all_file_name = sub_in_path + '\\' + file
            out_put = out_path + '\\' + file
            shutil.copyfile(all_file_name, out_put)

if __name__ == '__main__':
    all_train_path_1, all_train_path_0, all_test_path_1, all_test_path_0 = get_all_path(r'D:\医学数据集\breast\Breast_Histopathology_Images')
    CreateDir(r'D:\医学数据集\breast\Breast_Histopathology_Images\train_test\train\0')
    CreateDir(r'D:\医学数据集\breast\Breast_Histopathology_Images\train_test\test\1')
    CreateDir(r'D:\医学数据集\breast\Breast_Histopathology_Images\train_test\test\0')
    Copy_File(all_train_path_0, r'D:\医学数据集\breast\Breast_Histopathology_Images\train_test\train\0')
    Copy_File(all_test_path_0, r'D:\医学数据集\breast\Breast_Histopathology_Images\train_test\test\0')
    Copy_File(all_test_path_1, r'D:\医学数据集\breast\Breast_Histopathology_Images\train_test\test\1')