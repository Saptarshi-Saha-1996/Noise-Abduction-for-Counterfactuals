import pyro
import torch
import numpy as np
import torch.nn as nn

from torch.distributions import Independent
from pyro.infer.reparam.transform import TransformReparam
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.nn import PyroModule, pyro_method, DenseNN, AutoRegressiveNN
from pyro.distributions import Normal, Bernoulli, Categorical, TransformedDistribution, Uniform, RelaxedBernoulliStraightThrough
from pyro.distributions.transforms import Spline, AffineTransform, ExpTransform, ComposeTransform, spline_coupling, spline_autoregressive
from pyro.distributions.transforms import conditional_spline
from pyro.distributions.conditional import ConditionalTransformedDistribution
from .transforms import ConditionalAffineTransform, conditional_spline_autoregressive


class ConditionalSCM_Path(PyroModule):
    def __init__(
        self,
        age_dim=1,
        sex_dim=1,
        amount_dim=1,
        # risk_dim=1,
        context_dim=2,
        duration_dim=1,
        flow_dict=None,
        flow_type='spline',
        spline_bins=8,
        spline_order='linear',
        spline_hidden_dims=None,
        normalize=True
        ):
        super(ConditionalSCM_Path, self).__init__()
        self.sex_dim = sex_dim
        self.age_dim = age_dim
        self.amount_dim = amount_dim
        # self.risk_dim = risk_dim
        self.duration_dim= duration_dim
        self.context_dim = context_dim
        self.flow_dict = flow_dict
        self.flow_type = flow_type
        self.spline_bins = spline_bins
        self.spline_order = spline_order
        self.spline_hidden_dims = spline_hidden_dims
        self.normalize = normalize
        
        #Priors -------------------------------------------------------------------------------------------

        # sex prior
        self.sex_logits = torch.nn.Parameter(self.flow_dict['sex_logits'].cuda())
        
        # age priors
        #self.age_base_loc = torch.zeros([self.age_dim, ], device='cuda', requires_grad=False)
        #self.age_base_scale = torch.ones([self.age_dim, ], device='cuda', requires_grad=False)
        #self.register_buffer('age_flow_lognorm_loc', torch.zeros([], requires_grad=False))
        #self.register_buffer('age_flow_lognorm_scale', torch.ones([], requires_grad=False))
        
        # risk prior
        # self.risk_logits = torch.nn.Parameter(self.flow_dict['risk_logits'].cuda())
        
        # amount priors
        self.amount_base_loc = torch.zeros([self.amount_dim, ], device='cuda', requires_grad=False)
        self.amount_base_scale = torch.ones([self.amount_dim, ], device='cuda', requires_grad=False)
        self.register_buffer('amount_flow_lognorm_loc', torch.zeros([], requires_grad=False))
        self.register_buffer('amount_flow_lognorm_scale', torch.ones([], requires_grad=False))
        
        # duration priors
        self.duration_base_loc = torch.zeros([self.duration_dim, ], device='cuda', requires_grad=False)
        self.duration_base_scale = torch.ones([self.duration_dim, ], device='cuda', requires_grad=False)
        self.register_buffer('duration_flow_lognorm_loc', torch.zeros([], requires_grad=False))
        self.register_buffer('duration_flow_lognorm_scale', torch.ones([], requires_grad=False))
        

        
        #Flows ---------------------------------------------------------------------------------------------
        
        #We don't need to model age flows
        
        # age flows
        #self.age_flow_component = Spline(self.age_dim, count_bins=self.spline_bins)
        #self.age_flow_lognorm_loc = self.flow_dict['age_mean'].cuda()
        #self.age_flow_lognorm_scale = self.flow_dict['age_std'].cuda()
        #self.age_flow_normalize = AffineTransform(loc=self.age_flow_lognorm_loc.item(), scale=self.age_flow_lognorm_scale.item())
        #self.age_flow_constraint = ComposeTransform([self.age_flow_normalize, ExpTransform()])
        #self.age_flow_transforms = ComposeTransform([self.age_flow_component, self.age_flow_constraint])
        
        
        
        
        #amount flows 
        if self.flow_type == 'affine':
            amount_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
            self.amount_flow_component = ConditionalAffineTransform(context_nn=amount_net, event_dim=0)
        elif self.flow_type == 'spline':
            self.amount_flow_component = conditional_spline(self.amount_dim, context_dim=2, count_bins=self.spline_bins, order=self.spline_order)
        elif self.flow_type == 'autoregressive':
            self.amount_flow_component = conditional_spline_autoregressive(self.amount_dim, context_dim=2, hidden_dims=self.spline_hidden_dims, count_bins=self.spline_bins, order=self.spline_order)
            
        if self.normalize:
            self.amount_flow_lognorm_loc = self.flow_dict['amount_mean'].cuda()
            self.amount_flow_lognorm_scale = self.flow_dict['amount_std'].cuda()
            self.amount_flow_normalize = AffineTransform(loc=self.amount_flow_lognorm_loc.item(), scale=self.amount_flow_lognorm_scale.item())
            self.amount_flow_constraint = ComposeTransform([self.amount_flow_normalize, ExpTransform()])
            self.amount_flow_transforms = [self.amount_flow_component, self.amount_flow_constraint]
        else:
            self.amount_flow_transforms = ComposeTransformModule([self.amount_flow_component])
        
        
        # duration flows
        
        #if self.flow_type == 'affine':
            #duration_net = DenseNN(1, [8, 16], param_dims=[1,1],nonlinearity=torch.nn.LeakyReLU(.1))
            #self.duration_flow_component = ConditionalAffineTransform(context_nn=duration_net, event_dim=0)
        if self.flow_type == 'affine' or 'spline':
            self.duration_flow_component = conditional_spline(self.duration_dim, context_dim=1, count_bins=self.spline_bins, order=self.spline_order)
        elif self.flow_type == 'autoregressive':
            self.duration_flow_component = conditional_spline_autoregressive(self.duration_dim, context_dim=1, hidden_dims=self.spline_hidden_dims, count_bins=self.spline_bins, order=self.spline_order)
            
        if self.normalize:
            self.duration_flow_lognorm_loc = self.flow_dict['duration_mean'].cuda()
            self.duration_flow_lognorm_scale = self.flow_dict['duration_std'].cuda()
            self.duration_flow_normalize = AffineTransform(loc=self.duration_flow_lognorm_loc.item(), scale=self.duration_flow_lognorm_scale.item())
            self.duration_flow_constraint = ComposeTransform([self.duration_flow_normalize ,ExpTransform()])
            self.duration_flow_transforms = [self.duration_flow_component, self.duration_flow_constraint]
        else:
            self.duration_flow_transforms = ComposeTransformModule([self.duration_flow_component])
            
       
        # risk flow
#         if self.flow_type == 'affine':
#             risk_net = DenseNN(4, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
#             self.risk_flow_component = ConditionalAffineTransform(context_nn=risk_net, event_dim=0)
#         elif self.flow_type == 'spline':
#             self.risk_flow_component = conditional_spline(self.risk_dim, context_dim=4, count_bins=self.spline_bins, order=self.spline_order)
#         elif self.flow_type == 'autoregressive':
#             self.risk_flow_component = conditional_spline_autoregressive(self.risk_dim, context_dim=4, hidden_dims=self.spline_hidden_dims, count_bins=self.spline_bins, order=self.spline_order)
            
#         self.risk_flow_normalize = AffineTransform(loc=torch.tensor([0.7]).cuda() , scale=torch.tensor([2.5]).cuda())
#         self.risk_flow_constraint = ComposeTransform([self.risk_flow_normalize, ExpTransform()])
#         self.risk_flow_transforms = [self.risk_flow_component,self.risk_flow_constraint ]
        #ComposeTransformModule([self.risk_flow_normalize,self.risk_flow_component ])
        
            
            
            

    def pgm_model(self):
        
        # sex
        self.sex_dist = Categorical(logits=self.sex_logits).to_event(1)
        self.sex = pyro.sample('sex', self.sex_dist).cuda()
        #print('sex',self.sex.shape)
        
        # age
        #self.age_base_dist = Normal(self.age_base_loc, self.age_base_scale).to_event(1)
        #self.age_dist = TransformedDistribution(self.age_base_dist, self.age_flow_transforms)
        #self.age_dist=Normal(0,1).to_event(1)
        self.age = pyro.sample('age', Normal(0,1)).reshape(self.sex.shape).cuda()
        #print('age',self.age)
        #age_ = self.age_flow_constraint.inv(self.age)
        
        # amount
        context_amount = torch.cat([self.sex, self.age], -1)
        #print(context_amount)
        self.amount_base_dist = Normal(self.amount_base_loc.cuda(), self.amount_base_scale.cuda()).to_event(1)
        self.amount_dist = ConditionalTransformedDistribution(self.amount_base_dist, self.amount_flow_transforms)
        self.amount = pyro.sample('amount', self.amount_dist.condition(context_amount)).cuda()
        #print('amount',self.amount)
        
        # duration
        context_duration = torch.cat([self.amount], -1)
        self.duration_base_dist = Normal(self.duration_base_loc, self.duration_base_scale).to_event(1)
        self.duration_dist = ConditionalTransformedDistribution(self.duration_base_dist, self.duration_flow_transforms)
        self.duration = pyro.sample('duration', self.duration_dist.condition(context_duration))
        #print('duration',self.duration)
        
#         # risk
#         context_risk = torch.cat([self.sex, self.age,self.amount,self.duration], -1)
#         self.risk_base_dist= RelaxedBernoulliStraightThrough(temperature=torch.tensor(0.05).cuda(), probs=torch.tensor([0.4]).cuda()).to_event(1)
#         #Categorical(logits=self.risk_logits).to_event(1)
#         self.risk_dist = ConditionalTransformedDistribution(self.risk_base_dist, self.risk_flow_transforms).condition(context_risk)
        
#         self.risk= pyro.sample('risk', self.risk_dist)
        
        

    def forward(self, sex, age, amount, duration ):
        self.pgm_model()
        context_amount = torch.cat([sex,age], -1)
        context_duration = torch.cat([amount], -1)
        
        
        sex_logp = self.sex_dist.log_prob(sex)
        age_logp = torch.zeros(sex_logp.shape,device=sex_logp.get_device())
        amount_logp = self.amount_dist.condition(context_amount).log_prob(amount)
        duration_logp = self.duration_dist.condition(context_duration).log_prob(duration)

        
        return {'sex': sex_logp, 'age': age_logp, 'amount': amount_logp, 'duration': duration_logp}

    def clear(self):
        #self.sex_dist.clear_cache()
        #self.age_dist.clear_cache()
        self.amount_dist.clear_cache()
        self.duration_dist.clear_cache()
        # self.risk_dist.clear_cache()
        
    def model(self):
        self.pgm_model()
        return self.sex, self.age, self.amount, self.duration

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None
        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.model()
        return (*samples,)

    def sample_scm(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.scm()
        return (*samples,)

    def infer_exogeneous(self, **obs):
        cond_sample = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(obs['duration'].shape[0])
        #print(obs['duration'].shape[0])
        #print('cond_sample:-------\n',cond_sample)
        #print('cond_trace:-------\n',cond_trace)

        output = {}
        for name, node in cond_trace.nodes.items():
            if 'fn' not in node.keys():
                continue
            fn = node['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist
                
            if isinstance(fn, TransformedDistribution):
                output[name + '_base'] = ComposeTransform(fn.transforms).inv(node['value'])
                #print(name)
                #print( ComposeTransform(fn.transforms))
        return output

    def counterfactual(self, obs, condition, num_particles=1):
        counterfactuals = []
        for _ in range(num_particles):
            exogeneous = self.infer_exogeneous(**obs)
            if 'sex' not in condition.keys():
                exogeneous['sex'] = obs['sex']
            #print('observation---',obs)
            condition['age']=obs['age']
            #if 'scanner' not in condition.keys():
                #exogeneous['scanner'] = obs['scanner']

            counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogeneous), data=condition)(obs['duration'].shape[0])
            counterfactuals += [counter]
        return {k: v for k, v in zip(('sex', 'age', 'amount', 'duration'), (torch.stack(c).mean(0) for c in zip(*counterfactuals)))}


def _conditionalscm_path(arch, flow_dict, sex_dim, flow_type, bins, order):
    model = ConditionalSCM_Path(flow_dict=flow_dict, sex_dim=sex_dim,
        flow_type=flow_type, spline_bins=bins, spline_order=order)
    return model

def conditionalscm_path(flow_dict=None, sex_dim=1, flow_type='affine', bins=8, order='linear'):
    return _conditionalscm_path('conditionalscm_path', flow_dict, sex_dim, flow_type, bins, order)



##Saptarshi Saha