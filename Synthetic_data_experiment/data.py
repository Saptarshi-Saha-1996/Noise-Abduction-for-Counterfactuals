#import os
#import re
#import glob
#import time

#import torch.nn as nn
from torch.utils import data
import numpy as np
import torch

#import pandas as pd
#import scipy.misc as m
#from math import isnan, isinf

#import shutil
#import argparse
#import torch.optim as optim
#import torch.nn.functional as F
#import multiprocessing as mp

#from utils import mkdir



class featuresDataset(data.Dataset):

    def __init__(self,dim = 1, #noise=epsilon,
        inference = False, train = True,
        train_ratio = 0.8,
        data= None, sample_size=None,
        random_seed = None,
    ):
        super(featuresDataset, self).__init__()
        self.dim = dim
        self.random_seed =  random_seed
        self.inference = inference
        #self.noise=noise
        self.Data=data

        self.data, self.y = [], []
        #print('==> started preprocessing data ...')
        #X_6,X_5,X_4,X_3,X_2,X_1,Y= SCM(epsilon,False)
        #print(Y[0:5])
        for index in range(sample_size):
            self.data.append(dict(
                X6 =  np.asarray(self.Data['X6'][index]).astype(np.float32),
                X5 =  np.asarray(self.Data['X5'][index]).astype(np.float32),
                X4 =  np.asarray(self.Data['X4'][index]).astype(np.float32),
                X3 =  np.asarray(self.Data['X3'][index]).astype(np.float32),
                X2 =  np.asarray(self.Data['X2'][index]).astype(np.float32),
                X1 =  np.asarray(self.Data['X1'][index]).astype(np.float32),
                Y  =  np.asarray(self.Data['Y'][index]).astype(np.float32), 
                sample_id= index
            ))
        #print('==> finished preprocessing data ...')

        if not self.inference:                                                      #when inference is flase 
            self.ids = np.arange(0, len(self.data))           
            np.random.seed(random_seed)
            np.random.shuffle(self.ids)
            last_train_sample = int(len(self.ids) * train_ratio)
            if train:                                                               #when training is true
                self.ids = self.ids[:last_train_sample]
            else:
                self.ids = self.ids[last_train_sample:]
        else:                                                                       #when inference is true
            self.ids = np.arange(0, len(self.data))


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]
        info = self.data[id_]
        
        id= info['sample_id']
        X6 = torch.from_numpy(info['X6'])
        X5 = torch.from_numpy(info['X5'])
        X4 = torch.from_numpy(info['X4'])
        X3 = torch.from_numpy(info['X3'])
        X2 = torch.from_numpy(info['X2'])
        X1 = torch.from_numpy(info['X1'])
        Y = torch.from_numpy(info['Y'])
        

        if self.inference:
            return X6, X5, X4,  X3, X2,  X1, Y, id
        else:
            return X6, X5, X4,  X3, X2,  X1, Y


def get_features_dataset(
    data=None,
    dim=1,
    inference=False,
    train_ratio=0.8,
    sample_size=None,
    #noise=epsilon,
    random_seed=None,
    ):

    if not inference:
        dataset_train = featuresDataset(train=True, dim=dim,#noise=epsilon,
            train_ratio=train_ratio,data=data,sample_size=sample_size,random_seed=random_seed)
        dataset_test = featuresDataset(train=False,  dim=dim,#noise=epsilon,
            train_ratio=train_ratio,data=data,sample_size=sample_size, random_seed=random_seed)

        return dataset_train, dataset_test
    else:
        dataset = featuresDataset(inference=True,  dim=dim,#noise=epsilon,
            train_ratio=train_ratio,data=data,sample_size=sample_size, random_seed=random_seed)

        return dataset