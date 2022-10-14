import os
import re
import glob
import time
import torch
import numpy as np
import pandas as pd
import scipy.misc as m
from torch.utils import data
from math import isnan, isinf
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from torch.distributions.utils import probs_to_logits


class featuresDataset(data.Dataset):

    def __init__(
        self,
        filename,
        dim = 1,
        inference = False,
        train = True,
        train_ratio = 0.8,
        random_seed = 42,
        zscore = False
    ):
        super(featuresDataset, self).__init__()
        self.filename = filename
        self.data_path = os.path.join('./data', self.filename)
        self.data = pd.read_csv(self.data_path, low_memory=False)
        self.dim = dim
        self.random_seed = random_seed
        self.inference = inference
        self.zscore = zscore
        self.label_idx()
        self.flow_prepare()

        self.credit, self.y = [], []                                       #image_list---> credit
        #print('==> start preprocessing csv ...')
        for index, row in self.data.iterrows():
            self.credit.append(dict(
                duration = np.asarray(row['Duration']).astype(np.float32),
                amount = np.asarray(row['Credit amount']).astype(np.float32),
                sample_id=row['Index'],
                sex = self.sex_dict.transform([row['Sex']]),
                age = round(row['Age'], 0),                                ##round(Age,2) ----> round(Age,0)
            ))
        #print('==> finished preprocessing csv ...')

        if not self.inference:                                                      #when inference is flase 
            self.ids = np.arange(0, len(self.credit))           #img_idxes --> ids
            np.random.seed(self.random_seed)
            np.random.shuffle(self.ids)
            last_train_sample = int(len(self.ids) * train_ratio)
            if train:                                                               #when training is true
                self.ids = self.ids[:last_train_sample]
            else:
                self.ids = self.ids[last_train_sample:]
        else:                                                                       #when inference is true
            self.ids = np.arange(0, len(self.credit))

    def label_idx(self):
        self.sex_dict = LabelEncoder()
        self.sex_dict.fit(self.data.Sex.tolist())
        self.sex_class = len(self.sex_dict.classes_)
        self.covariates_dict = {'sex': self.sex_dict}

    def flow_prepare(self):
        # sex
        sex_dict = LabelEncoder()
        sex_dict.fit(self.data.Sex.tolist())
        sex = sex_dict.transform(self.data.Sex.tolist())
        sex_counts = Counter(sex)
        sex_mass = [v/sum(sex_counts.values()) for k,v in sex_counts.items()]
        sex_logits = probs_to_logits(torch.as_tensor(sex_mass), is_binary=False)
        
        # age
        age_mean = torch.as_tensor(self.data.Age.to_numpy()).log().mean()
        age_std = torch.as_tensor(self.data.Age.to_numpy()).log().std()
        
        
        # amount
        amount_mean=torch.as_tensor(self.data['Credit amount'].to_numpy()).log().mean()
        amount_std=torch.as_tensor(self.data['Credit amount'].to_numpy()).log().std()
        
        #duration
        duration_mean=torch.as_tensor(self.data['Duration'].to_numpy()).log().mean()
        duration_std=torch.as_tensor(self.data['Duration'].to_numpy()).log().std()
        

        self.flow_dict = {'sex_logits': sex_logits.unsqueeze(0).float(),
                          'age_mean': age_mean.float(),
                          'age_std': age_std.float(),
                          'amount_mean': amount_mean.float(),
                          'amount_std': amount_std.float(),
                          'duration_mean':duration_mean.float(),
                          'duration_std':duration_std.float()
                         }

    def zScoreNorm(self, x, min_max=True):
        x=x.astype(np.float32)
        if min_max:
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        else:
            res = (x - np.min(x)) / np.max(x)
            res = (res * 1.0 - np.mean(res)) / np.std(res)
        return res

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]
        info = self.credit[id_]
        duration_ = info['duration']
        amount_=info['amount']

        if self.zscore:
            duration_ = self.zScoreNorm(duration_)
            amount_ = self.zScoreNorm(amount_)
        
        
        id= info['sample_id']
        duration = torch.from_numpy(duration_)
        amount = torch.from_numpy(amount_)
        sex = torch.tensor(info['sex']).float()
        age = torch.tensor(info['age']).unsqueeze(-1).float()
        

        if self.inference:
            return sex,  age, amount,  duration, id
        else:
            return sex,  age, amount,  duration


def get_features_dataset(
    filename,
    dim=1,
    inference=False,
    train_ratio=0.8,
    random_seed=42,
    zscore=False
    ):

    if not inference:
        dataset_train = featuresDataset(train=True, filename=filename, dim=dim,
            train_ratio=train_ratio, random_seed=random_seed, zscore=zscore)
        dataset_test = featuresDataset(train=False, filename=filename, dim=dim,
            train_ratio=train_ratio, random_seed=random_seed, zscore=zscore)

        return dataset_train, dataset_test
    else:
        dataset = featuresDataset(inference=True, filename=filename, dim=dim,
            train_ratio=train_ratio, random_seed=random_seed, zscore=zscore)

        return dataset

    
if __name__=='__main__':
    x=get_features_dataset('german_credit_data.csv',inference=True)
    print(x[12])
    print(x.flow_dict)
    print("----------------------")
    print(x.credit[0])
    print(x.credit[1])

    
    
    
    
#Saptarshi Saha 