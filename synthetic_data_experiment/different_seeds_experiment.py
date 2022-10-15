import numpy as np
import torch
import pyro



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pyro_seed= 68

from config import sample_size,epochs
from scm import SCM
from full_model import conditionalscm
from partial_model import conditionalscm_partial
from train import train_model 

from infer import do_inference
from analysis import error_estimate ,sampling_capabilities, noise_infer_capabilities
from analysis import plot_Y_cf
from cf_analysis import cf_analysis
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os


experiment_name='different_seeds_experiment'

np.random.seed(1729)
intervention_set=np.random.uniform(-30,30,200)

seeds=[1729, 9, 83,689,251 ,558,12,5692,23, 7]
pyro_seed= 68

print('seeds:',seeds)
print('sample_size',sample_size)

time_full=[]
time_partial=[]

for seed in seeds:
    print("seed no:",seed,'-----------------------------Start')
    np.random.seed(seed)
    torch.manual_seed(seed)
    #pyro.set_rng_seed(pyro_seed)
    epsilon=np.random.normal(0,1,(sample_size,7))
    X_6,X_5,X_4,X_3,X_2,X_1,Y= SCM(epsilon=epsilon,intervention=False)
    Data={'X6':torch.tensor(X_6).reshape(-1,1).float(),'X5':torch.tensor(X_5).reshape(-1,1).float(),'X4':torch.tensor(X_4).reshape(-1,1).float(),'X3':torch.tensor(X_3).reshape(-1,1).float(),'X2':torch.tensor(X_2).reshape(-1,1).float(),'X1':torch.tensor(X_1).reshape(-1,1).float(),'Y':torch.tensor(Y).reshape(-1,1).float()}
   
    scm=conditionalscm()
    scm_partial=conditionalscm_partial()
    
    pyro.set_rng_seed(pyro_seed)
    logs=experiment_name + '/logs/seed_'+str(seed)
    if not os.path.exists(logs):
        os.makedirs(logs)
    
    time_fullmodel=train_model(scm,logs,seed=seed,data=Data,print_mode=False)
    time_full.append(time_fullmodel)
    time_partialmodel=train_model(scm_partial,logs,seed=seed,data=Data,print_mode=False)
    time_partial.append(time_partialmodel)
    j= 5
    intervention='do(X5='+str(j)+')'
    X6_cf,X5_cf, X4_cf, X3_cf, X2_cf, X1_cf, Y_cf= SCM(epsilon,True,j)
    
    #logs='./'+save
    SCM_samples,SCM_inferred_noise,SCM_counterfactuals=                       do_inference(scm,intervention,save=logs,data=Data,print_mode=False)
    SCM_partial_samples,SCM_partial_inferred_noise,SCM_partial_counterfactuals=do_inference(scm_partial,intervention,save=logs,data=Data,print_mode=False)
    
    path=experiment_name+'/assets/seed_'+str(seed)
    if not os.path.exists(path):
        os.makedirs(path)
        
    path=path+'/'
    sampling_capabilities(Data=Data,SCM_samples=SCM_samples,SCM_partial_samples=SCM_partial_samples,path=path)
    noise_infer_capabilities(epsilon=epsilon,SCM_partial_inferred_noise=SCM_partial_inferred_noise,SCM_inferred_noise=SCM_inferred_noise,path=path )
    plot_Y_cf(Y_cf=Y_cf,j=j,SCM_partial_counterfactuals=SCM_partial_counterfactuals,SCM_counterfactuals=SCM_counterfactuals,path=path)
    error_fullmodel,error_partialmodel=cf_analysis(intervention_set=intervention_set,path=logs,data=Data,SCM=SCM,scm=scm,scm_partial=scm_partial,epsilon=epsilon)
    
    if seed==seeds[0]:
        cf_error=pd.DataFrame({'intervention':intervention_set,'error_partial':error_partialmodel,'error_full': error_fullmodel,'seed':seed})
    else:
        cf_error2=pd.DataFrame({'intervention':intervention_set,'error_partial':error_partialmodel,'error_full': error_fullmodel,'seed':seed})
        cf_error=cf_error.append(cf_error2,ignore_index=True)
        
    np.random.seed(seed+1)
    
    epsilon_unseen=np.random.normal(0,1,(sample_size,7))
    X_6,X_5,X_4,X_3,X_2,X_1,Y= SCM(epsilon_unseen,False)
    Data_unseen={'X6':torch.tensor(X_6).reshape(-1,1).float(),'X5':torch.tensor(X_5).reshape(-1,1).float(),'X4':torch.tensor(X_4).reshape(-1,1).float(),'X3':torch.tensor(X_3).reshape(-1,1).float(),'X2':torch.tensor(X_2).reshape(-1,1).float(),'X1':torch.tensor(X_1).reshape(-1,1).float(),'Y':torch.tensor(Y).reshape(-1,1).float()}
    unseenerror_fullmodel,unseenerror_partialmodel=cf_analysis(intervention_set=intervention_set,data=Data_unseen,path=logs,SCM=SCM,scm=scm,scm_partial=scm_partial,epsilon=epsilon_unseen)
    if seed==seeds[0]:
        unseen_cf_error=pd.DataFrame({'intervention':intervention_set,'error_partial':unseenerror_partialmodel,'error_full': unseenerror_fullmodel,'seed':seed})
    else:
        unseen_cf_error2=pd.DataFrame({'intervention':intervention_set,'error_partial':unseenerror_partialmodel,'error_full': unseenerror_fullmodel,'seed':seed})
        unseen_cf_error=unseen_cf_error.append(unseen_cf_error2,ignore_index=True)
    
    print("seed no:",seed,'--------------------------------Completed')



training_time=pd.DataFrame({'full_model':time_full,'partial_model':time_partial,'seed':seeds})
filepath=experiment_name+'/training_time.csv'
training_time.to_csv(filepath)

filepath_=experiment_name+'/cf_error.csv'
cf_error.to_csv(filepath_)

from matplotlib import pyplot as plt
import seaborn as sns

#sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(8, 4))
#plt.scatter(intervention_set,np.exp(np.array(error_partial)),label='partial model',color='black',ls='-',marker='o')
#plt.scatter(intervention_set,np.exp(np.array(error_full)),label='full model',color='yellow',ls='--',marker='*')
sns.lineplot(data=cf_error,x='intervention',y='error_partial',label='partial model',color='black',marker='o',alpha=0.5,ci='sd')
sns.lineplot(data=cf_error,x='intervention',y='error_full',label='full model',color='green',marker='.',alpha=0.5,ci='sd')

plt.xlabel(r'Intervention value on $X_{5}$ ',fontweight='bold')
plt.ylabel('mean squared error',fontweight='bold')
plt.legend()
fig.savefig(experiment_name+'/counterfactual_errors.pdf',format='pdf',pad_inches=0.1,bbox_inches='tight',dpi=1200)


filepath_=experiment_name+'/unseen_cf_error.csv'
unseen_cf_error.to_csv(filepath_)
unseen_cf_error


#sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(8, 4))
#plt.scatter(intervention_set,np.exp(np.array(error_partial)),label='partial model',color='black',ls='-',marker='o')
#plt.scatter(intervention_set,np.exp(np.array(error_full)),label='full model',color='yellow',ls='--',marker='*')
sns.lineplot(data=unseen_cf_error,x='intervention',y='error_partial',label='partial model',color='black',marker='o',alpha=0.5,ci='sd')
sns.lineplot(data=unseen_cf_error,x='intervention',y='error_full',label='full model',color='green',marker='.',alpha=0.5,ci='sd')

plt.xlabel(r'Intervention value on $X_{5}$ ',fontweight='bold')
plt.ylabel('mean squared error',fontweight='bold')
plt.legend()
fig.savefig(experiment_name+'/counterfactual_unseen_errors.pdf',format='pdf',pad_inches=0.1,bbox_inches='tight',dpi=1200)

print('COMPLETED')