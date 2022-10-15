import numpy as np


def SCM(epsilon=None, intervention=False, *args):
    X_4=2*epsilon[:,4]+1
    X_6=epsilon[:,6]-1
    if not intervention:
        X_5=3*X_6+epsilon[:,5]-1
    else:
        X_5=np.full(X_6.shape[0],args)
    X_2=X_5-epsilon[:,2]
    X_3=-3*X_4+epsilon[:,3]-3
    X_1=X_6-X_5+3*epsilon[:,1]
    Y=X_1+2*X_2-3*X_3+epsilon[:,0]
    return X_6,X_5,X_4,X_3,X_2,X_1,Y