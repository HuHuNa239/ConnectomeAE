# 导入使用到的库
import csv, os, random
import numpy as np
import SimpleITK as sitk
import torch
# from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import scipy.io
import torch

# 调试时的参数加载


'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要加载 __init__，__len__和__getitem__方法
'''
Threholds = 0.8 # 0.8
Threholdf = 0.0 # 0.5


# 为训练数据 创建数据加载子类
class train_Dataset(Data.Dataset):
    def __init__(self, root, fold):
        super(train_Dataset, self).__init__()
        # 初始化
        self.root = root
        self.fold = fold
        # self.i=i

        # 通过CSV的形式获取图像与标签
        # self.images, self.labels = self.load_csv(str(self.fold) + '_train.csv')
        # 再次打乱数据引入随机性
        smat, s_feamat, fmat, labels = self.load_csv(str(self.fold) + '_train.csv')
        pairs = [[a, b, c, e] for a, b, c, e in zip(smat, s_feamat, fmat, labels)]
        random.shuffle(pairs)
        self.smat = [i[0] for i in pairs]
        self.s_feamat = [i[1] for i in pairs]
        self.fmat = [i[2] for i in pairs]
        self.labels = [i[3] for i in pairs]

    # 将所有的图像与标签路径配对好写入CSV
    def load_csv(self, filename):
        smat, s_feamat, fmat, labels = [], [], [], []
        with open(os.path.join(self.root, str(self.fold), filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                sbasemat, sfeamat, fbasemat,  label = row[0], row[1], row[2], row[3]
                label = int(label)
                smat.append(sbasemat)
                s_feamat.append(sfeamat)
                fmat.append(fbasemat)
                labels.append(label)
        assert len(labels) == len(smat) == len(s_feamat) == len(fmat)
        return smat, s_feamat, fmat, labels

    def __len__(self):
        # 返回数据集的大小
        return len(self.smat)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        smat, s_feamat, fmat, label = self.smat[index], self.s_feamat[index],  self.fmat[index], self.labels[index]


        def buildmat(x2, x1, x3):
            stopology = x1
            stopology = scipy.io.loadmat(stopology)
            stopology = stopology['average_pearson']
            stopology[stopology < Threholds] = 0


            sfeamat = scipy.io.loadmat(x2)
            sfeamat = sfeamat['baseline_feature']
            # sfea_mat = np.array(sfeamat)

            ftopology = x3
            ftopology = scipy.io.loadmat(ftopology)
            ftopology = ftopology['average_pearson']
            ftopology[ftopology < Threholdf] = 0

            return sfeamat, stopology, ftopology

        smat, stopology, ftopology = buildmat(smat, s_feamat, fmat)

        label = torch.tensor(label)
        stopology = torch.from_numpy(stopology)
        ftopology = torch.from_numpy(ftopology)
        smat = torch.from_numpy(smat)

        return stopology, smat, ftopology, label




# 为验证数据 创建数据加载子类
class val_Dataset(Data.Dataset):
    def __init__(self, root, fold):
        super(val_Dataset, self).__init__()
        # 初始化
        self.root = root
        self.fold = fold

        # 通过CSV的形式获取图像与标签
        # self.images, self.labels = self.load_csv(str(self.fold) + '_train.csv')
        # 再次打乱数据引入随机性
        smat, sfeamat, fmat, labels = self.load_csv(str(self.fold) + '_val.csv')
        pairs = [[a, b, c, d] for a, b, c, d in zip(smat, sfeamat, fmat, labels)]
        random.shuffle(pairs)
        self.smat = [i[0] for i in pairs]
        self.sfeamat = [i[1] for i in pairs]
        self.fmat = [i[2] for i in pairs]
        self.labels = [i[3] for i in pairs]

    # 将所有的图像与标签路径配对好写入CSV
    def load_csv(self, filename):
        smat, s_feamat, fmat, labels = [], [], [], []
        with open(os.path.join(self.root, str(self.fold), filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                sbasemat, sfeamat, fbasemat, label = row[0], row[1], row[2], row[3]
                label = int(label)
                smat.append(sbasemat)
                s_feamat.append(sfeamat)
                fmat.append(fbasemat)
                labels.append(label)
        assert len(labels) == len(smat) == len(s_feamat) == len(fmat)
        return smat, s_feamat, fmat, labels

    def __len__(self):
        # 返回数据集的大小
        return len(self.smat)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        smat, sfeamat, fmat, label = self.smat[index], self.sfeamat[index], self.fmat[index], self.labels[index]


        def buildmat(x2, x1, x3):
            stopology = x1
            stopology = scipy.io.loadmat(stopology)
            stopology = stopology['average_pearson']
            stopology[stopology < Threholds] = 0

            sfeamat = scipy.io.loadmat(x2)
            sfeamat = sfeamat['baseline_feature']
            # sfea_mat = np.array(sfeamat)

            ftopology = x3
            ftopology = scipy.io.loadmat(ftopology)
            ftopology = ftopology['average_pearson']
            ftopology[ftopology < Threholdf] = 0

            return sfeamat, stopology, ftopology

        smat, stopology, ftopology = buildmat(smat, sfeamat, fmat)

        # 标签加载
        label = torch.tensor(label)
        stopology = torch.from_numpy(stopology)
        ftopology = torch.from_numpy(ftopology)
        smat = torch.from_numpy(smat)

        return stopology, smat, ftopology, label

# 为测试数据 创建数据加载子类
class test_Dataset(Data.Dataset):
    def __init__(self, root, fold):
        super(test_Dataset, self).__init__()
        # 初始化
        self.root = root
        self.fold = fold

        # 通过CSV的形式获取图像与标签
        # self.images, self.labels = self.load_csv(str(self.fold) + '_train.csv')
        # 再次打乱数据引入随机性
        smat, sfeamat, fmat, labels = self.load_csv(str(self.fold) + '_test.csv')
        pairs = [[a, b, c, d] for a, b, c, d in zip(smat, sfeamat, fmat, labels)]
        random.shuffle(pairs)
        self.smat = [i[0] for i in pairs]
        self.sfeamat = [i[1] for i in pairs]
        self.fmat = [i[2] for i in pairs]
        self.labels = [i[3] for i in pairs]

    # 将所有的图像与标签路径配对好写入CSV
    def load_csv(self, filename):
        smat, s_feamat, fmat, labels = [], [], [], []
        with open(os.path.join(self.root, str(self.fold), filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                sbasemat, sfeamat, fbasemat, label = row[0], row[1], row[2], row[3]
                label = int(label)
                smat.append(sbasemat)
                s_feamat.append(sfeamat)
                fmat.append(fbasemat)
                labels.append(label)
        assert len(labels) == len(smat) == len(s_feamat) == len(fmat)
        return smat, s_feamat, fmat, labels

    def __len__(self):
        # 返回数据集的大小
        return len(self.smat)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        smat, sfeamat, fmat, label = self.smat[index], self.sfeamat[index], self.fmat[index], self.labels[index]


        def buildmat(x2, x1, x3):
            stopology = x1
            stopology = scipy.io.loadmat(stopology)
            stopology = stopology['average_pearson']
            stopology[stopology < Threholds] = 0

            sfeamat = scipy.io.loadmat(x2)
            sfeamat = sfeamat['baseline_feature']
            # sfea_mat = np.array(sfeamat)

            ftopology = x3
            ftopology = scipy.io.loadmat(ftopology)
            ftopology = ftopology['average_pearson']
            ftopology[ftopology < Threholdf] = 0

            return sfeamat, stopology, ftopology

        smat, stopology, ftopology = buildmat(smat, sfeamat, fmat)

        # 标签加载
        label = torch.tensor(label)
        stopology = torch.from_numpy(stopology)
        ftopology = torch.from_numpy(ftopology)
        smat = torch.from_numpy(smat)

        return stopology, smat, ftopology, label
