import sys
from tkinter import image_names
sys.path.append("..")
import numpy as np 
import os 
import torch.utils.data as data
import pandas as pd 
import numpy as np 

import xlrd 
import matplotlib.pyplot as plt 

import  clip 
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from scipy import stats
import scipy.io as sio
import random
import torchvision
from PIL import Image

from six.moves import cPickle as pickle #for performance


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB') #RGB')
def pil_loader1(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

class DHD(data.Dataset):
    def __init__(self, root, index, transform, transform1, patch_num, training):
        self.root = root
        mat_file = os.path.join(root, 'MOS.mat')
        data = sio.loadmat(mat_file)
        labels = data['MOS'] # [[0],...[1]]

        choose_index = []
        start_ind = 1
        for ind in index:
            for i in range(7):
                choose_index.append(start_ind + ind * 7 + i) 
        sample = []
        for idx in choose_index:
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'Dehaze', '%04d.png'%idx), labels[idx-1][0])) #ds[idx-1][0]

        self.samples = sample
        self.transform = transform
        self.training = training
        self.transform1 = transform1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index] #  ds 
        sample = pil_loader(path)

        if self.transform is not None:
            Local = self.transform(sample)
            Global = self.transform1(sample)
        return Local, target/100.0, path, Global 

    def __len__(self):
        length = len(self.samples)
        return length

class exBeDDE(data.Dataset):
    def __init__(self, root, index, transform, patch_num, training, cache=True):
        self.root = root
        # range : 0- 1
        dirname = ['beijing', 'changsha', 'chengdu', 'hangzhou', 'hefei', 'hongkong', 'lanzhou','nanchang', 'shanghai', 'shenyang', 'tianjing', 'wuhan']
        
        choose_index = []
        for ind in index:
            choose_index.append(dirname[ind]) 

        sample = []
        for idx in choose_index:
            rootdir = os.path.join(root, idx)
            for dn in os.listdir(rootdir):
                if idx in dn:
                    imgdir = os.path.join(rootdir, dn)
                    mat_file = os.path.join(imgdir, '%s_scores.mat'%dn)
                    data = sio.loadmat(mat_file)['imageScores']

                    imgname = data['image_names'][0][0]
                    score = data['scores'][0][0]
                    for img, s in zip(imgname.tolist(), score):
                        img = img[0][0]
                        s = s[0]
                        for aug in range(patch_num):
                            sample.append((os.path.join(imgdir, img), s)) #ds[idx-1][0]

        self.samples = sample
        self.transform = transform
        self.training = training
        if cache:
            self.cache_sample = {}
            self.cache = cache

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index] #  ds 

        sample = 0
        if self.cache:
            if self.cache_sample.get(path, None) is None:
                sample = pil_loader(path)
                self.cache_sample[path] = sample 
            else:
                sample = self.cache_sample[path]
        else:
            sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path, ds

    def __len__(self):
        length = len(self.samples)
        return length

if __name__ == '__main__':
    root = '/home/fujun/datasets/iqa/DHD'
    ds = DHD(root, [0, 1], torchvision.transforms.ToTensor(), 1,1)
    root = '/home/fujun/datasets/iqa/exBeDDE'
    ds = exBeDDE(root, [0, 1], torchvision.transforms.ToTensor(), 1,1)
    ds[0]
