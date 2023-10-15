#coding:utf8
import os
from torch.utils import data
from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def filter_data(data,label, no = 2):
    use_index = [idx for idx,val in enumerate(label) if val != no]
    max_label,min_label = max(label[use_index]),min(label[use_index])
    select_label = label[use_index]

    select_label[select_label==min_label]=0
    select_label[select_label==max_label]=1
    return data[use_index], select_label

class AllData_DataFrame(data.Dataset):

    def __init__(self, args, is_train = False, is_val = False, is_test = False):
        self.dataset = args.dataset
        self.no = args.no
        self.fold = args.fold
        self.type = args.type

        self.is_train = is_train
        self.is_val = is_val
        self.is_test = is_test

        self.is_mixup = True

        if self.dataset == "AD":
            all_data = loadmat(f"./dataset/process/cv-{args.fold}/cm-{args.complete}/NC_MCI_AD.mat")
        elif self.dataset == "PD":
            all_data = loadmat(f"./dataset/process/cv-{args.fold}/cm-{args.complete}/NC_RBD_PD.mat")
        else:
            print(self.dataset)
            exit()

        if self.is_train is True:
            c_train_data = all_data['c_train_data']
            u1_train_data = all_data['1u_train_data'][...,0]
            u2_train_data = all_data['2u_train_data'][...,1]
            c_train_label = all_data['c_train_label'][0]
            u1_train_label = all_data['1u_train_label'][0]
            u2_train_label = all_data['2u_train_label'][0]

            c_train_data, c_train_label = filter_data(c_train_data, c_train_label, self.no)
            u1_train_data, u1_train_label = filter_data(u1_train_data, u1_train_label, self.no)
            u2_train_data, u2_train_label = filter_data(u2_train_data, u2_train_label, self.no)


            c_train_data, c_train_label = filter_data(c_train_data, c_train_label, self.no)
            u1_train_data, u1_train_label = filter_data(u1_train_data, u1_train_label, self.no)
            u2_train_data, u2_train_label = filter_data(u2_train_data, u2_train_label, self.no)

            if self.type == 'c':
                self.length = len(c_train_data)
                self.imgs = [c_train_data]
                self.lbls = [c_train_label]

            elif self.type == 'adv':
                # inferred 
                all_data = np.load(f"./adv/{args.complete}/{args.dataset}_{args.fold}_{args.no}.npz")
                c_train_data = all_data['data'].transpose(0,2,3,1)
                c_train_label = all_data['label']
                print(c_train_data.shape)
                self.length = len(c_train_data)
                self.imgs = [c_train_data]
                self.lbls = [c_train_label]

            elif  self.type =='mean':
                self.length = len(c_train_data) + len(u1_train_data) + len(u2_train_data)
                self.imgs = [c_train_data, u1_train_data,u2_train_data]
                self.lbls = [c_train_label, u1_train_label, u2_train_label]
                self.get_mean()

            elif self.type == 'knn' or self.type =='mixup':

                self.length = len(c_train_data) + len(u1_train_data) + len(u2_train_data)
                self.imgs = [c_train_data, u1_train_data,u2_train_data]
                self.lbls = [c_train_label, u1_train_label, u2_train_label]
                if self.type == 'knn':
                    self.knn_prepare()

        elif self.is_val is True:
            c_test_data = all_data['c_val_data']
            c_test_label = all_data['c_val_label'][0]
            c_test_data, c_test_label = filter_data(c_test_data, c_test_label, self.no)

            self.length = len(c_test_data)
            self.imgs = [c_test_data]
            self.lbls = [c_test_label]
            
        else:
            c_test_data = all_data['c_test_data']
            c_test_label = all_data['c_test_label'][0]
            c_test_data, c_test_label = filter_data(c_test_data, c_test_label, self.no)

            self.length = len(c_test_data)
            self.imgs = [c_test_data]
            self.lbls = [c_test_label]
        print("data length: ",self.length)

    def mixup(self, index):
        c_data, u1_data, u2_data = self.imgs
        c_label, u1_label, u2_label = self.lbls

        len1 = len(c_data)
        len2 = len(c_data) + len(u1_data)
        len3 = len(c_data) + len(u1_data) + len(u2_data)
        
        if index < len1:
            img1 = c_data[index][...,0]
            img2 = c_data[index][...,1]
            label = c_label[index]

        elif index < len2:
            img1 = u1_data[index - len1]
            label = u1_label[index - len1]

            if self.is_mixup is True:
                u2_index = [i for i in range(len(u2_data)) if u2_label[i] == label]
                np.random.shuffle(u2_index)
                img2 = u2_data[u2_index[0]]
        else:
            img2 = u2_data[index - len2]
            label = u2_label[index - len2]

            if self.is_mixup is True:
                u1_index = [i for i in range(len(u1_data)) if u1_label[i] == label]
                np.random.shuffle(u1_index)
                img1 = u1_data[u1_index[0]]
        return img1, img2, label

    def get_test(self, index):
        c_data = self.imgs[0]
        c_label = self.lbls[0]

        img1 = c_data[index][...,0]
        img2 = c_data[index][...,1]
        label = c_label[index]
        return img1, img2, label

    def get_mean(self):
        c_data, u1_data, u2_data = self.imgs
        c_label, u1_label, u2_label = self.lbls

        pos_index = u2_label[u2_label==0]
        pos_mean_val = u2_data[pos_index].mean(0)

        neg_index = u2_label[u2_label==1]
        neg_mean_val = u2_data[neg_index].mean(0)

        new_u1 = []
        for dat, lab in zip(u1_data,u1_label):
            if lab == 0:
                new_data = np.concatenate([dat[None,...,None],pos_mean_val[None,...,None]], 3)
            elif lab == 1:
                new_data = np.concatenate([dat[None,...,None],neg_mean_val[None,...,None]], 3)
            else:
                print("error lab")
                exit()
            new_u1.append(new_data)
        new_u1 = np.array(new_u1)[:,0]

        pos_index = u1_label[u1_label==0]
        pos_mean_val = u1_data[pos_index].mean(0)

        neg_index = u1_label[u1_label==1]
        neg_mean_val = u1_data[neg_index].mean(0)
        new_u2 = []

        for dat, lab in zip(u2_data,u2_label):
            if lab == 0:
                new_data = np.concatenate([pos_mean_val[None,...,None],dat[None,...,None]], 3)
            elif lab == 1:
                new_data = np.concatenate([neg_mean_val[None,...,None],dat[None,...,None]], 3)
            new_u2.append(new_data)

        new_u2 = np.array(new_u2)[:,0]

        print('mean data: ', new_u1.shape, new_u2.shape, c_data.shape)
        self.imgs = np.concatenate([c_data, new_u1, new_u2], 0)
        self.lbls = np.concatenate([c_label, u1_label, u2_label], 0)

    def get_complete(self, index):
        c_data, c_label = self.imgs[0], self.lbls[0]
        img1 = c_data[index][..., 0]
        img2 = c_data[index][..., 1]
        return img1, img2, c_label[index]

    def knn_prepare(self):
        c_data, u1_data, u2_data = self.imgs
        c_label, u1_label, u2_label = self.lbls

        mask = np.zeros((100,100))
        for i in range(100):
            for j in range(i+1, 100):
                mask[i,j] = 1

        knn1 = KNeighborsClassifier()
        knn2 = KNeighborsClassifier()
        knn1.fit(c_data[:,mask==1, 0], c_label)
        knn2.fit(c_data[:,mask==1, 1], c_label)
        num_neighbor = 1#len(c_data) #1

        ## training set1
        neighbor = knn1.kneighbors(u1_data[:,mask==1],num_neighbor,False)

        supple_train_u1 = []
        for i in range(len(neighbor)):
            indexs = neighbor[i]
            tmp = np.mean([c_data[...,1][index] for index in indexs], 0)
            supple_train_u1.append(tmp)
        supple_train_u1 = np.array(supple_train_u1)
        u1_c_data = np.concatenate([u1_data[...,None], supple_train_u1[...,None]], 3)

        ## training set2
        num_neighbor = 1 #len(c_data) #1
        neighbor = knn2.kneighbors(u2_data[:,mask==1],num_neighbor,False)

        supple_train_u2 = []
        for i in range(len(neighbor)):
            indexs = neighbor[i]
            tmp = np.mean([c_data[...,0][index] for index in indexs], 0)
            supple_train_u2.append(tmp)
        supple_train_u2 = np.array(supple_train_u2)
        u2_c_data = np.concatenate([u2_data[...,None], supple_train_u2[...,None]], 3)

        self.all_train_data = np.concatenate([c_data, u1_c_data, u2_c_data], 0)
        self.all_train_label = np.concatenate([c_label, u1_label, u2_label], 0)
    def knn(self, index):
        return self.all_train_data[index,...,0], self.all_train_data[index,...,1], self.all_train_label[index]
    def __getitem__(self,index):
        
        if self.is_train is True:
            if self.type == 'c' or self.type == 'adv':
                img1, img2, label = self.get_complete(index)

            elif self.type == 'mixup':
                img1, img2, label = self.mixup(index)

            elif self.type == 'knn':
                img1, img2, label = self.knn(index)
            elif self.type == 'mean':
                img1 = self.imgs[index][..., 0]
                img2 = self.imgs[index][..., 1]
                label = self.lbls[index]
        else:
            img1, img2, label = self.get_test(index)
        img1 = img1[np.newaxis,...]
        img2 = img2[np.newaxis,...]
        return img1, img2, label
    
    def __len__(self):
        return self.length
