
import os
import re
import glob
import time
import torch
import pyro

import torch.nn as nn
from torch.utils import data
import numpy as np

import pandas as pd
import scipy.misc as m
from math import isnan, isinf

import shutil
import argparse
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing as mp

from torch.distributions import Independent
from pyro.infer.reparam.transform import TransformReparam
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.nn import PyroModule, pyro_method
from pyro.distributions import Normal,TransformedDistribution
from pyro.distributions.transforms import Spline,affine_coupling,conditional_affine_coupling, affine_autoregressive,conditional_affine_autoregressive, AffineTransform, ExpTransform, ComposeTransform, spline_coupling 
from pyro.distributions.transforms import conditional_spline
from pyro.distributions.conditional import ConditionalTransformedDistribution

from utils import mkdir






class ConditionalSCM(PyroModule):
    def __init__(self,context_dim=2,normalize=True,spline_bins=8,spline_order='linear'):
        super(ConditionalSCM, self).__init__()
        self.context_dim = context_dim
        self.spline_bins = spline_bins
        self.spline_order = spline_order
        self.normalize = normalize
        

        
        #Flows ---------------------------------------------------------------------------------------------
        
        # X_4 flows
        self.X4_flow_component =  affine_coupling(1)                         #Spline(1, count_bins=self.spline_bins)
        self.X4_flow_transforms = ComposeTransform([self.X4_flow_component])
        
        # X_6 flows
        self.X6_flow_component =   affine_coupling(1)                        #Spline(1, count_bins=self.spline_bins)
        self.X6_flow_transforms = ComposeTransform([self.X6_flow_component])
        
        #X_5 flows
        self.X5_flow_component =   conditional_affine_coupling(input_dim=1, context_dim=1,hidden_dims=[10,10])  #conditional_spline(1, context_dim=1, count_bins=self.spline_bins, order=self.spline_order)
        self.X5_flow_transforms = ComposeTransformModule([self.X5_flow_component])
        
        #X_2 flows 
        self.X2_flow_component =     conditional_affine_coupling(input_dim=1, context_dim=1,hidden_dims=[10,10]) #conditional_spline(1, context_dim=1, count_bins=self.spline_bins, order=self.spline_order)
        self.X2_flow_transforms = ComposeTransformModule([self.X2_flow_component])
        
        #X_3 flows 
        self.X3_flow_component =     conditional_affine_coupling(input_dim=1, context_dim=1,hidden_dims=[10,10])                         #conditional_spline(1, context_dim=1, count_bins=self.spline_bins, order=self.spline_order)
        self.X3_flow_transforms = ComposeTransformModule([self.X3_flow_component])
        
        
        #X_1 flows 
        self.X1_flow_component =     conditional_affine_coupling(input_dim=1, context_dim=2,hidden_dims=[10,10])          #conditional_spline(1, context_dim=2, count_bins=self.spline_bins, order=self.spline_order)
        self.X1_flow_transforms = ComposeTransformModule([self.X1_flow_component])
        
        #Y flows
        self.Y_flow_component =  conditional_affine_coupling(input_dim=1, context_dim=3,hidden_dims=[10,10])                                        #conditional_spline(1, context_dim=3, count_bins=self.spline_bins, order=self.spline_order)
        self.Y_flow_transforms = ComposeTransformModule([self.Y_flow_component])
        
       

            
    def pgm_model(self,**kwargs):
        loc=torch.tensor([0.0])
        scale=torch.tensor([1.0])
        
        #observed={}
        #if kwargs:
        #    for key, value in kwargs.items():
        #        observed[key]=value
        #else:
        #    observed={'X4':None,'X6':None,'X5':None,'X3':None,'X2':None,'X1':None,'Y':None}
        
        
        
        
        
        # X4
        self.X4_base_dist = Normal(loc,scale).to_event(1)
        self.X4_dist=TransformedDistribution(self.X4_base_dist, self.X4_flow_transforms)
        self.X4 = pyro.sample('X4', self.X4_dist,) #obs=observed['X4']
        # X6
        self.X6_base_dist = Normal(loc,scale).to_event(1)
        self.X6_dist=TransformedDistribution(self.X6_base_dist, self.X6_flow_transforms)
        self.X6 = pyro.sample('X6', self.X6_dist,) #obs=observed['X6']
        #X5
        context_X5=torch.cat([self.X6],-1)
        self.X5_base_dist = Normal(loc,scale).to_event(1)
        self.X5_dist = ConditionalTransformedDistribution(self.X5_base_dist, self.X5_flow_transforms)
        self.X5 = pyro.sample('X5', self.X5_dist.condition(context_X5),) #obs=observed['X5']
        #X3
        context_X3=torch.cat([self.X4],-1)
        self.X3_base_dist = Normal(loc,scale).to_event(1)
        self.X3_dist = ConditionalTransformedDistribution(self.X3_base_dist, self.X3_flow_transforms)
        self.X3 = pyro.sample('X3', self.X3_dist.condition(context_X3),) #obs=observed['X3']
        #X2
        context_X2=torch.cat([self.X5],-1)
        self.X2_base_dist = Normal(loc,scale).to_event(1)
        self.X2_dist = ConditionalTransformedDistribution(self.X2_base_dist, self.X2_flow_transforms)
        self.X2 = pyro.sample('X2', self.X2_dist.condition(context_X2),) #obs=observed['X2']
        #X1
        context_X1=torch.cat([self.X5,self.X6],-1)
        #print('context_X1',context_X1)
        self.X1_base_dist = Normal(loc,scale).to_event(1)
        self.X1_dist = ConditionalTransformedDistribution(self.X1_base_dist, self.X1_flow_transforms)
        self.X1 = pyro.sample('X1', self.X1_dist.condition(context_X1),) #obs=observed['X1']
        # Y
        context_Y=torch.cat([self.X3,self.X2,self.X1],-1)
        self.Y_base_dist = Normal(loc,scale).to_event(1)
        self.Y_dist = ConditionalTransformedDistribution(self.Y_base_dist, self.Y_flow_transforms)
        self.Y = pyro.sample('Y', self.Y_dist.condition(context_Y),) #obs=observed['Y']
       
        
        

    def forward(self, X6,X5, X4,X3,X2,X1,Y ):
        self.pgm_model() #**{"X6":X6,"X5": X5,"X4": X4,"X3":X3,"X2":X2,"X1":X1,"Y":Y})
        
        
                
 
        
        
        context_X5=torch.cat([X6],-1).detach()
        context_X3=torch.cat([X4],-1).detach()
        context_X2=torch.cat([X5],-1).detach()
        context_X1=torch.cat([X5,X6],-1).detach()
        context_Y=torch.cat([X3,X2,X1],-1).detach()
         
        X6_logp = self.X6_dist.log_prob(X6)
        X5_logp = self.X5_dist.condition(context_X5).log_prob(X5)
        X4_logp = self.X4_dist.log_prob(X4)
        X3_logp = self.X3_dist.condition(context_X3).log_prob(X3)
        X2_logp = self.X2_dist.condition(context_X2).log_prob(X2)
        X1_logp = self.X1_dist.condition(context_X1).log_prob(X1)
        Y_logp = self.Y_dist.condition(context_Y).log_prob(Y)
        
        return {'X_6':X6_logp,'X_5':X5_logp,'X_4': X4_logp, 'X_3': X3_logp, 'X_2': X2_logp, 'X_1': X1_logp,'Y': Y_logp}

    def clear(self):
        self.X5_dist.clear_cache()
        self.X3_dist.clear_cache()
        self.X2_dist.clear_cache()
        self.X1_dist.clear_cache()
        self.Y_dist.clear_cache()
        
    def model(self):
        self.pgm_model()
        return self.X6, self.X5, self.X4, self.X3, self.X2, self.X1, self.Y

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None
        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample(self, n_samples=1):
        with pyro.plate('sample_observations', n_samples):
            samples = self.model()
        return (*samples,)

    def sample_scm(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.scm()
        return (*samples,)
    
    def infer_exogeneous(self, **obs):
        cond_sample = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(obs['X1'].shape[0])
        output = {}
        for name, node in cond_trace.nodes.items():
            if 'fn' not in node.keys():
                continue
            fn = node['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist

            if isinstance(fn, TransformedDistribution):
                output[name + '_base'] = ComposeTransform(fn.transforms).inv(node['value'])
        return output

    def counterfactual(self, obs,intervention, num_particles=1):
        counterfactuals = []

        for _ in range(num_particles):
            exogeneous = self.infer_exogeneous(**obs)
            condition= {intervention.split('=')[0].split('(')[-1] :torch.full_like( obs['X4'],float(intervention.split('=')[-1][:-1]))}
            counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogeneous), data=condition)(obs['X1'].shape[0]) 
            counterfactuals += [counter]
            #print('helper',helper)
            #print('helper',helper)

        return {k: v for k, v in zip(('X_6','X_5','X_4', 'X_3', 'X_2', 'X_1','Y'), (torch.stack(c).mean(0) for c in zip(*counterfactuals)))}


def conditionalscm():
    model = ConditionalSCM(context_dim=2,normalize=True,spline_bins=16,spline_order='linear')
    return model
