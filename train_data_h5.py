import os.path
import h5py
import math
import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import ToTensor
from scipy.io import loadmat
import pandas as pd
def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]

def get_h5py(path, save_path):
    file_h5f = h5py.File(save_path, 'w')
    print(os.path.exists(path))
    list_i = os.listdir(path)
    list_i.sort()
    for i in range(len(list_i)):
        img = ToTensor()(Image.open(os.path.join(path, list_i[i])))
        data = img.numpy()
        patch = file_h5f.create_dataset(str(i), data=data[:, :, :])
    return file_h5f

def get_mat(path, save_path):
    file_h5f = h5py.File(save_path, 'w')
    list_i = os.listdir(path)
    list_i.sort()
    for i in range(len(list_i)):
        mat =loadmat(os.path.join(path, list_i[i]))
        data = np.array(mat["conditionmat"])
        patch = file_h5f.create_dataset(str(i), data=data[:, :, :])
    return file_h5f

def prepare_data(data_path):
    # train
    print('process training data')
    print(os.path.exists(data_path))
    input_h_path = os.path.join(data_path, 'input_h')
    center_path = os.path.join(data_path, 'center')
    gt_path = os.path.join(data_path, 'gt')
    conditionmat_path = os.path.join(data_path, 'conditionmat')
    condition_path = os.path.join(data_path, 'condition')
    depth_path = os.path.join(data_path, 'depth')

    save_input_h_path = os.path.join(data_path, 'input_h.h5')
    save_center_path = os.path.join(data_path, 'center.h5')
    save_gt_path = os.path.join(data_path, 'gt.h5')
    save_conditionmat_path = os.path.join(data_path, 'conditionmat.h5')
    save_condition_path = os.path.join(data_path, 'condition.h5')
    save_depth_path = os.path.join(data_path, 'depth.h5')

    input_h_h5f = get_h5py(input_h_path, save_input_h_path)
    center_h5f = get_h5py(center_path, save_center_path)
    gt_h5f = get_h5py(gt_path, save_gt_path)
    conditionmat_h5f = get_mat(conditionmat_path, save_conditionmat_path)
    condition_h5f = get_h5py(condition_path, save_condition_path)
    depth_h5f = get_h5py(depth_path, save_depth_path)

    input_h_h5f.close()
    center_h5f.close()
    gt_h5f.close()
    conditionmat_h5f.close()
    condition_h5f.close()
    depth_h5f.close()

def prepare_data_val(data_path):
    # train
    print('process training data')
    input_h_path = os.path.join(data_path, 'input_h')
    center_path = os.path.join(data_path, 'center')
    gt_path = os.path.join(data_path, 'gt')
    conditionmat_path = os.path.join(data_path, 'conditionmat')
    condition_path = os.path.join(data_path, 'condition')
    depth_path = os.path.join(data_path, 'depth')

    save_input_h_path = os.path.join(data_path, 'input_h.h5')
    save_center_path = os.path.join(data_path, 'center.h5')
    save_gt_path = os.path.join(data_path, 'gt.h5')
    save_conditionmat_path = os.path.join(data_path, 'conditionmat.h5')
    save_condition_path = os.path.join(data_path, 'condition.h5')
    save_depth_path = os.path.join(data_path, 'depth.h5')

    input_h_h5f = get_h5py(input_h_path, save_input_h_path)
    center_h5f = get_h5py(center_path, save_center_path)
    gt_h5f = get_h5py(gt_path, save_gt_path)
    conditionmat_h5f = get_mat(conditionmat_path, save_conditionmat_path)
    condition_h5f = get_h5py(condition_path, save_condition_path)
    depth_h5f = get_h5py(depth_path, save_depth_path)

    input_h_h5f.close()
    center_h5f.close()
    gt_h5f.close()
    conditionmat_h5f.close()
    condition_h5f.close()
    depth_h5f.close()
    
def prepare_data_test(data_path):
    # train
    print('process training data')
    input_h_path = os.path.join(data_path, 'input_h')
    center_path = os.path.join(data_path, 'center')
    conditionmat_path = os.path.join(data_path, 'conditionmat')
    condition_path = os.path.join(data_path, 'condition')
    depth_path = os.path.join(data_path, 'depth')

    save_input_h_path = os.path.join(data_path, 'input_h.h5')
    save_center_path = os.path.join(data_path, 'center.h5')
    save_conditionmat_path = os.path.join(data_path, 'conditionmat.h5')
    save_condition_path = os.path.join(data_path, 'condition.h5')
    save_depth_path = os.path.join(data_path, 'depth.h5')

    input_h_h5f = get_h5py(input_h_path, save_input_h_path)
    center_h5f = get_h5py(center_path, save_center_path)
    conditionmat_h5f = get_mat(conditionmat_path, save_conditionmat_path)
    condition_h5f = get_h5py(condition_path, save_condition_path)
    depth_h5f = get_h5py(depth_path, save_depth_path)

    input_h_h5f.close()
    center_h5f.close()
    conditionmat_h5f.close()
    condition_h5f.close()
    depth_h5f.close()

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, train_data_dir):
        super().__init__()
        self.path = train_data_dir
    def __getitem__(self, index):
        input_h_path = os.path.join(self.path, 'input_h.h5')
        center_path = os.path.join(self.path, 'center.h5')
        gt_path = os.path.join(self.path, 'gt.h5')
        conditionmat_path = os.path.join(self.path, 'conditionmat.h5')
        condition_path = os.path.join(self.path, 'condition.h5')
        depth_path = os.path.join(self.path, 'depth.h5')

        input_h_h5f = h5py.File(input_h_path, 'r')
        center_h5f = h5py.File(center_path, 'r')
        gt_h5f = h5py.File(gt_path, 'r')
        conditionmat_h5f = h5py.File(conditionmat_path, 'r')
        condition_h5f = h5py.File(condition_path, 'r')
        depth_h5f = h5py.File(depth_path, 'r')

        input_h= input_h_h5f[str(index)][:]
        center= center_h5f[str(index)][:]
        gt= gt_h5f[str(index)][:]
        conditionmat= conditionmat_h5f[str(index)][:]
        condition= condition_h5f[str(index)][:]
        depth= depth_h5f[str(index)][:]

        input_h_h5f.close()
        center_h5f.close()
        gt_h5f.close()
        conditionmat_h5f.close()
        condition_h5f.close()
        depth_h5f.close()
        return input_h, center, gt, conditionmat, condition, depth

    def __len__(self):
        list_i = os.listdir(os.path.join(self.path, 'input_h'))
        return int(len(list_i))

class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        self.path = val_data_dir
    def __getitem__(self, index):
        input_h_path = os.path.join(self.path, 'input_h.h5')
        center_path = os.path.join(self.path, 'center.h5')
        gt_path = os.path.join(self.path, 'gt.h5')
        conditionmat_path = os.path.join(self.path, 'conditionmat.h5')
        condition_path = os.path.join(self.path, 'condition.h5')
        depth_path = os.path.join(self.path, 'depth.h5')

        input_h_h5f = h5py.File(input_h_path, 'r')
        center_h5f = h5py.File(center_path, 'r')
        gt_h5f = h5py.File(gt_path, 'r')
        conditionmat_h5f = h5py.File(conditionmat_path, 'r')
        condition_h5f = h5py.File(condition_path, 'r')
        depth_h5f = h5py.File(depth_path, 'r')

        input_h= input_h_h5f[str(index)][:]
        center= center_h5f[str(index)][:]
        gt= gt_h5f[str(index)][:]
        conditionmat= conditionmat_h5f[str(index)][:]
        condition= condition_h5f[str(index)][:]
        depth= depth_h5f[str(index)][:]

        input_h_h5f.close()
        center_h5f.close()
        gt_h5f.close()
        conditionmat_h5f.close()
        condition_h5f.close()
        depth_h5f.close()
        return input_h, center, gt, conditionmat, condition, depth

    def __len__(self):
        list_i = os.listdir(os.path.join(self.path, 'condition'))
        return int(len(list_i))

class TestData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        self.path = val_data_dir
    def __getitem__(self, index):
        input_h_path = os.path.join(self.path, 'input_h.h5')
        center_path = os.path.join(self.path, 'center.h5')
        conditionmat_path = os.path.join(self.path, 'conditionmat.h5')
        condition_path = os.path.join(self.path, 'condition.h5')
        depth_path = os.path.join(self.path, 'depth.h5')

        input_h_h5f = h5py.File(input_h_path, 'r')
        center_h5f = h5py.File(center_path, 'r')
        conditionmat_h5f = h5py.File(conditionmat_path, 'r')
        condition_h5f = h5py.File(condition_path, 'r')
        depth_h5f = h5py.File(depth_path, 'r')

        input_h= input_h_h5f[str(index)][:]
        center= center_h5f[str(index)][:]
        conditionmat= conditionmat_h5f[str(index)][:]
        condition= condition_h5f[str(index)][:]
        depth= depth_h5f[str(index)][:]

        input_h_h5f.close()
        center_h5f.close()
        conditionmat_h5f.close()
        condition_h5f.close()
        depth_h5f.close()
        return input_h, center, conditionmat, condition, depth

    def __len__(self):
        list_i = os.listdir(os.path.join(self.path, 'input_h'))
        return int(len(list_i))