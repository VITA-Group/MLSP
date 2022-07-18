import torch
import torch.utils.data as data
import os
import sys
import h5py
import numpy as np
from multiprocessing.dummy import Pool
from torchvision import transforms
import glob
import random
import threading
import time
from utils.metapc_utils import *
from utils.pc_utils import (farthest_point_sample_np, scale_to_unit_cube, jitter_pointcloud,
                            rotate_shape, random_rotate_one_axis)
# from data_utils import *

class PaddingData(data.Dataset):
    def __init__(self, io, pc_root, aug=False, partition='train', pc_input_num=1024, swapax=False): #density=0, drop=0, p_scan=0, 
        super(PaddingData, self).__init__()

        # self.status = status
        self.partition = partition

        self.pc_list = []
        self.lbl_list = []
        # self.density = density
        # self.drop = drop
        # self.p_scan = p_scan
        if partition == 'train':
            npy_list = glob.glob(os.path.join(pc_root, '*', 'train', '*.npy'))
        else:
            npy_list = glob.glob(os.path.join(pc_root, '*', 'test', '*.npy'))
        categorys = glob.glob(os.path.join(pc_root, '*'))
        categorys = [c.split(os.path.sep)[-1] for c in categorys]
        categorys = sorted(categorys)
        print(categorys)
        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(categorys.index(_dir.split('/')[-3]))

        self.label = np.asarray(self.lbl_list)
        # print('self.label',self.label)
        # print('npy_list',self.pc_list)
        self.num_examples = len(self.pc_list)
        self.swapax = swapax
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 12 < 10]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 12 >= 10]).astype(np.int)
            np.random.shuffle(self.val_ind)
        
        io.cprint("number of " + partition + " examples in " + str(pc_root) + ": " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in " + str(pc_root) + partition + " set: " + str(dict(zip(unique, counts))))


        self.aug = aug
        
        self.transforms = transforms.Compose(
            [
                PointcloudToTensor(),
                PointcloudScale(),
                PointcloudRotate(),
                PointcloudRotatePerturbation(),
                PointcloudTranslate(),
                PointcloudJitter(),
            ]
        )
        self.pc_input_num = pc_input_num

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        if self.swapax:
            pointcloud[:, 1] = pointcloud[:, 2] + pointcloud[:, 1]
            pointcloud[:, 2] = pointcloud[:, 1] - pointcloud[:, 2]
            pointcloud[:, 1] = pointcloud[:, 1] - pointcloud[:, 2]            
        label = np.copy(self.label[item])
        pointcloud = scale_to_unit_cube(pointcloud)
        if self.aug:
            pointcloud = self.transforms(pointcloud)
            pointcloud = pointcloud.numpy()

        if pointcloud.shape[0] > self.pc_input_num:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, self.pc_input_num)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
        elif pointcloud.shape[0] < self.pc_input_num:
            pointcloud = np.append(pointcloud, np.zeros((self.pc_input_num - pointcloud.shape[0], 3)), axis=0)
            pointcloud = pointcloud[:self.pc_input_num]

        return (pointcloud, label)

    def __len__(self):
        return len(self.pc_list)