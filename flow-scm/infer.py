import os
import re
import time
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing as mp
import pyro

import models
from utils import mkdir
from loaders import get_features_dataset

#---------------------------------------------------------------------------------------

class FeaturesInference():
    def __init__(self, args, model, loader,  original_data, covariates_dict):
        super(FeaturesInference, self).__init__()
        self.args = args
        self.model = model
        self.loader = loader
        self.dim = args.dim
        self.name=args.model_path.split('/')[-2]
        self.original_df = original_data
        self.covariates_dict = covariates_dict
        self.output_path = os.path.join('./datasets', 'counterfactuals')
        mkdir(self.output_path)
        self.key = re.sub(' ', '', re.sub('[(=,)]', '_', args.condition))[:-1]
        #print(self.model)
        #print('path',self.name)
        self.output_file = os.path.join(self.output_path, self.name+'_' +self.key+'.csv')
        # args.data_filename[:-4] + '_' + self.key +'_'+str(self.model.__class__.__name__).lower()
        self.counter_df = pd.DataFrame(columns=self.original_df.columns)
        #print(self.counter_df)

#---------------------------------------------
    def update_csv(self, samples, ids):
        amount = samples['amount'].cpu().numpy()
        sex = samples['sex'].cpu().numpy().squeeze()
        age = samples['age'].cpu().numpy().squeeze()
        duration = samples['duration'].cpu().numpy().squeeze()
        #print('ids',ids)
        
        for idx in range(len(ids)):
            #print('Check',self.original_df.Index==ids[idx].numpy())
            #print('-------')
            self.counter_df = self.counter_df.append(self.original_df[self.original_df.Index == ids[idx].numpy()], ignore_index=True)
            #print('Saha',self.counter_df.index[-1])
            self.counter_df.loc[self.counter_df.index[-1],'Duration'] = np.round(duration[idx],0)
            self.counter_df.loc[self.counter_df.index[-1], 'Sex'] = self.covariates_dict['sex'].inverse_transform([int(sex[idx])])[-1]
            self.counter_df.loc[self.counter_df.index[-1], 'Age'] = age[idx]
            self.counter_df.loc[self.counter_df.index[-1], 'Credit amount'] = np.round(amount[idx],0)

    def close_csv(self):
        # recover
        #raw_df = self.original_df.copy()
        #har_df = self.counter_df.copy()
        #img = raw_df.iloc[:, -self.feature_dim:].to_numpy()
        #min = np.repeat(np.expand_dims(np.min(img, axis=1), axis=1), self.feature_dim, axis=1)
        #max = np.repeat(np.expand_dims(np.max(img, axis=1), axis=1), self.feature_dim, axis=1)
        #self.counter_df.iloc[:, -self.feature_dim:] = har_df.iloc[:, -self.feature_dim:] * (max - min) + min # recover back from standardize
        # save
        self.counter_df.to_csv(self.output_file, index=False)
        print('\n=> Conterfactuals has been stored at {}.'.format(self.output_file.split('/')[-1]))

#------------------------------------------------------------------------------------------------------        
    def counterfactual_conditions(self, data):
        counterfactuals = {
            'do(sex=Female)': {'sex': torch.zeros_like(data['sex'])},
            'do(sex=Male)': {'sex': torch.ones_like(data['sex']) * 1}}
        return counterfactuals

    def inference(self):
        for batch_idx, (sex,  age, amount,  duration, id) in enumerate(self.loader):
            with torch.no_grad():
                if self.args.cuda:
                    sex,  age, amount,  duration = sex.reshape(-1,1).cuda(),  age.reshape(-1,1).cuda(), amount.reshape(-1,1).cuda(),  duration.reshape(-1,1).cuda()
                data = {'amount': amount, 'sex': sex, 'age': age, 'duration': duration}
                #print(data)
                condition = self.counterfactual_conditions(data)[self.args.condition]
                samples = self.model.counterfactual(obs=data, condition=condition, num_particles=self.args.particles)
                #print(samples,id)
                self.update_csv(samples, id)
                #if batch_idx % self.args.log_interval == 0:
                    #print('Inference: [{}/{} ({:.1f}%)]\tCondition: {}'.format(
                        #batch_idx * len(duration), len(self.loader.dataset), 100. * batch_idx / len(self.loader), self.args.condition))
        self.close_csv()

#-----------------------------------------------------------------------------------------------------------

def main(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    dataset = get_features_dataset(
        inference=True, filename=args.data_filename, dim=args.dim, random_seed=args.data_seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    model = models.__dict__[args.arch](
    flow_dict=dataset.flow_dict, flow_type=args.flow_type, order=args.flow_order)
    checkpoint = torch.load(args.model_path)
    print(args.model_path)
    print('==> Best LogProb: {:.6f}\n'.format(checkpoint['best_loss']))
    print('associative power',checkpoint['associative power'])
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    if args.cuda:
        model.cuda()

    feature = FeaturesInference(args, model, loader, dataset.data, dataset.covariates_dict)
    start_time = time.time()
    feature.inference()
    print('Time: {:.2f} min\n'.format((time.time()-start_time)/60.))
    print('Done...\n\n\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Flow SCM')
    parser.add_argument('--data-filename', default='features_data.csv', type=str, metavar='PATH',
                    help='dataset csv file name')
    parser.add_argument('--dim', type=int, default=1,
                    help='dimension of the data features (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default='0',
                    help='gpu id')
    parser.add_argument('--data-seed', type=int, default=69, metavar='S',
                    help='dataset seed (default: 69)')
    parser.add_argument('--seed', type=int, default=9, metavar='S',
                    help='random seed (default: 9s)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save model (default: current directory)')
    parser.add_argument('--arch', default='conditionalscm', type=str,
                    help='architecture to use')
    parser.add_argument('--flow-type', default='autoregressive', type=str,
                    choices=['affine', 'spline', 'autoregressive'],
                    help='type of flow to use')
    parser.add_argument('--flow-order', default='linear', type=str,
                    choices=['linear', 'quadratic'], help='order of flow to use')
    parser.add_argument('--particles', default=32, type=int,
                    help='number of particles for sampling (default: 32)')
    parser.add_argument('--condition', default='do(sex=Male)', type=str,
                    choices=['do(sex=Male)', 'do(sex=Female)'],
                    help='counterfactuals condition')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.set_device('cuda:' + args.gpu_id)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        pyro.set_rng_seed(args.seed)

    args.save = os.path.join(args.save, args.arch + '_flowtype_' + args.flow_type
        + '_floworder_' + args.flow_order)
    args.model_path = os.path.join(args.save, 'flow_best.pth.tar')

    main(args)

##Saptarshi Saha