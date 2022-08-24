from infer import do_inference
from config import sample_size
import numpy as np

def cf_analysis(intervention_set=None,path=None,data=None,SCM=None,scm=None,scm_partial=None,epsilon=None):

    error_full=[]
    error_partial=[]
    
    for j in intervention_set:
        intervention='do(X5='+str(j)+')'
        SCM_samples,SCM_inferred_noise,SCM_counterfactuals=do_inference(scm,intervention,save=path,data=data) #noise=epsilon)
            
        SCM_partial_samples,SCM_partial_inferred_noise,SCM_partial_counterfactuals=do_inference(scm_partial,intervention,save=path,data=data)#noise=epsilon)
        X6_cf, X5_cf, X4_cf, X3_cf, X2_cf, X1_cf, Y_cf= SCM(epsilon,True,j)
    
        error_full.append( np.linalg.norm(Y_cf.reshape(-1)-SCM_counterfactuals['Y'].detach().numpy().reshape(-1))/(sample_size))
        error_partial.append(np.linalg.norm(Y_cf.reshape(-1)-SCM_partial_counterfactuals['Y'].detach().numpy().reshape(-1))/(sample_size))

    return error_full, error_partial
        