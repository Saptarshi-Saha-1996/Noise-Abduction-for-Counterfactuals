import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from config import sample_size


def error_estimate(inferred_noise,epsilon):
    dict={}
    for key,value in inferred_noise.items():
        t=key.split('_')[0][-1]
        if t=='Y':
            s=0
        else:
            s=int(t)
        dict[key.split('_')[0]+' error']=np.linalg.norm(inferred_noise[key].detach().numpy().reshape(-1)-epsilon[:,s])/sample_size
    return dict



def sampling_capabilities(Data=None,SCM_samples=None,SCM_partial_samples=None,path=None):
    sample_Y={'True':Data['Y'].reshape(-1),'full':SCM_samples['Y'].detach().numpy().reshape(-1),'partial':SCM_partial_samples['Y'].detach().numpy().reshape(-1)}
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(30,6))
    sns.kdeplot(sample_Y['True'], ax=ax1, color='red',legend=True,label='True',alpha=0.2,fill=True)
    sns.kdeplot(sample_Y['full'], ax=ax1,color='green',legend=True,label='full model',alpha=0.8,linewidth=3,linestyle='-')
    sns.kdeplot(sample_Y['partial'], ax=ax1,color='black',legend=True,label='partial model',alpha=0.8,linewidth=3,linestyle='-.')
    ax1.legend()
    ax1.set_ylabel(' ',fontweight='bold')
    ax1.set_title(r'$Y$',fontsize=15)

    sample_X1={'True':Data['X1'].reshape(-1),'full':SCM_samples['X1'].detach().numpy().reshape(-1),'partial':SCM_partial_samples['X1'].detach().numpy().reshape(-1)}
    sns.kdeplot(sample_X1['True'], ax=ax2, color='red',legend=True,label='True',alpha=0.2,fill=True)
    sns.kdeplot(sample_X1['full'], ax=ax2,color='green',legend=True,label='full model',alpha=0.8,linewidth=3,linestyle='-')
    sns.kdeplot(sample_X1['partial'], ax=ax2,color='black',legend=True,label='partial model',alpha=0.8,linewidth=3,linestyle='-.')
    ax2.legend()
    ax2.set_ylabel(' ',fontweight='bold')
    ax2.set_title(r'$X_{1}$',fontsize=15)


    sample_X2={'True':Data['X2'].reshape(-1),'full':SCM_samples['X2'].detach().numpy().reshape(-1),'partial':SCM_partial_samples['X2'].detach().numpy().reshape(-1)}
    sns.kdeplot(sample_X2['True'], ax=ax3, color='red',legend=True,label='True',alpha=0.2,fill=True)
    sns.kdeplot(sample_X2['full'], ax=ax3,color='green',legend=True,label='full model',alpha=0.8,linestyle='-',linewidth=3)
    sns.kdeplot(sample_X2['partial'], ax=ax3,color='black',legend=True,label='partial model',alpha=0.8,linestyle='-.',linewidth=3)
    ax3.legend()
    ax3.set_ylabel(' ',fontweight='bold')
    ax3.set_title(r'$X_{2}$',fontsize=15)
    fig.savefig(path+'sampling_capabilities.pdf',pad_inches=0.1,bbox_inches='tight',format='pdf',dpi=600)

    
    
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(1,4,figsize=(30, 6))
    sample_X3={'True':Data['X3'].reshape(-1),'full':SCM_samples['X3'].detach().numpy().reshape(-1)}
    sns.kdeplot(sample_X3['True'], ax=ax0, color='red',legend=True,label='True',alpha=0.2,fill=True)
#sns.histplot(sample_Y['partial'], ax=ax1,color='black',legend=True,label='partial model',alpha=0.2,kde=True)
    sns.kdeplot(sample_X3['full'], ax=ax0,color='green',legend=True,label='full model',alpha=1,linewidth=3,linestyle='-')
    ax0.legend()
    ax0.set_ylabel(' ',fontweight='bold')
    ax0.set_title(r'$X_{3}$',fontsize=15)



    sample_X4={'True':Data['X4'].reshape(-1),'full':SCM_samples['X4'].detach().numpy().reshape(-1)}
    sns.kdeplot(sample_X4['True'], ax=ax1, color='red',legend=True,label='True',alpha=0.2,fill=True)
#sns.histplot(sample_Y['partial'], ax=ax1,color='black',legend=True,label='partial model',alpha=0.2,kde=True)
    sns.kdeplot(sample_X4['full'], ax=ax1,color='green',legend=True,label='full model',alpha=1,linewidth=3,linestyle='-')
    ax1.legend()
    ax1.set_ylabel(' ',fontweight='bold')
    ax1.set_title(r'$X_{4}$',fontsize=15)

    sample_X5={'True':Data['X5'].reshape(-1),'full':SCM_samples['X5'].detach().numpy().reshape(-1)}
    sns.kdeplot(sample_X5['True'], ax=ax2, color='red',legend=True,label='True',alpha=0.2,fill=True)
#sns.histplot(sample_X5['partial'], ax=ax2,color='black',legend=True,label='partial model',alpha=0.2,kde=True)
    sns.kdeplot(sample_X5['full'], ax=ax2,color='green',legend=True,label='full model',alpha=1,linewidth=3,linestyle='-')
    ax2.legend()
    ax2.set_ylabel(' ',fontweight='bold')
    ax2.set_title(r'$X_{5}$',fontsize=15)


    sample_X6={'True':Data['X6'].reshape(-1),'full':SCM_samples['X6'].detach().numpy().reshape(-1)}
    sns.kdeplot(sample_X6['True'], ax=ax3, color='red',legend=True,label='True',alpha=0.2,fill=True)
#sns.histplot(sample_X6['partial'], ax=ax3,color='black',legend=True,label='partial model',alpha=0.2,kde=True)
    sns.kdeplot(sample_X6['full'], ax=ax3,color='green',legend=True,label='full model',alpha=1,linewidth=3,linestyle='-')
    ax3.legend()
    ax3.set_ylabel(' ',fontweight='bold')
    ax3.set_title(r'$X_{6}$',fontsize=15)
    fig.savefig(path+'additional_sampling_capabilities.pdf',pad_inches=0.1,bbox_inches='tight',format='pdf',dpi=600)
    

def noise_infer_capabilities(epsilon=None,SCM_partial_inferred_noise=None,SCM_inferred_noise=None,path=None ):
    infer_noise_Y={'True':epsilon[:,0].reshape(-1),'partial':SCM_partial_inferred_noise['Y_base'].detach().numpy().reshape(-1),'full':SCM_inferred_noise['Y_base'].detach().numpy().reshape(-1)}
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(30, 6))
    sns.kdeplot(infer_noise_Y['True'], ax=ax1, color='red',legend=True,label='True',alpha=0.2,fill=True)
    sns.kdeplot(infer_noise_Y['full'], ax=ax1,color='green',legend=True,label='full model',alpha=1,linewidth=2,linestyle='-')
    sns.kdeplot(infer_noise_Y['partial'], ax=ax1,color='black',legend=True,label='partial model',alpha=1,linewidth=2,linestyle='-.')
    ax1.legend()
    ax1.set_ylabel(' ',fontweight='bold')
    ax1.set_title(r'$\epsilon_{Y}$',fontsize=15)

    infer_noise_X1={'True':epsilon[:,1].reshape(-1),'partial':SCM_partial_inferred_noise['X1_base'].detach().numpy().reshape(-1),'full':SCM_inferred_noise['X1_base'].detach().numpy().reshape(-1)}
    sns.kdeplot(infer_noise_X1['True'], ax=ax2, color='red',legend=True,label='True',alpha=0.2,fill=True)
    sns.kdeplot(infer_noise_X1['full'], ax=ax2,color='green',legend=True,label='full model',alpha=1,linewidth=3,linestyle='-')
    sns.kdeplot(infer_noise_X1['partial'], ax=ax2,color='black',legend=True,label='partial model',alpha=1,linewidth=3,linestyle='-.')
    ax2.legend()
    ax2.set_ylabel(' ',fontweight='bold')
    ax2.set_title(r'$\epsilon_{1}$',fontsize=15)


    infer_noise_X2={'True':epsilon[:,2].reshape(-1),'partial':SCM_partial_inferred_noise['X2_base'].detach().numpy().reshape(-1),'full':SCM_inferred_noise['X2_base'].detach().numpy().reshape(-1)}
    sns.kdeplot(infer_noise_X2['True'], ax=ax3, color='red',legend=True,label='True',alpha=0.2,fill=True)
    sns.kdeplot(infer_noise_X2['full'], ax=ax3,color='green',legend=True,label='full model',alpha=1,linewidth=3,linestyle='-')
    sns.kdeplot(infer_noise_X2['partial'], ax=ax3,color='black',legend=True,label='partial model',alpha=1,linewidth=3,linestyle='-.')
    ax3.legend()
    ax3.set_ylabel(' ',fontweight='bold')
    ax3.set_title(r'$\epsilon_{2}$',fontsize=15)


    fig.savefig(path+'inferred_noises.pdf',pad_inches=0.1,bbox_inches='tight',format='pdf',dpi=600)
    
    
    infer_noise_X3={'True':epsilon[:,3].reshape(-1),'full':SCM_inferred_noise['X3_base'].detach().numpy().reshape(-1)}
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(30, 5))
    sns.kdeplot(infer_noise_X3['True'], ax=ax1, color='red',legend=True,label='True',alpha=0.2,fill=True)
#sns.histplot(infer_noise_Y['partial'], ax=ax1,color='black',legend=True,label='partial model',alpha=0.2,kde=True)
    sns.kdeplot(infer_noise_X3['full'], ax=ax1,color='green',legend=True,label='full model',alpha=1,linewidth=3,linestyle='-')
    ax1.legend()
    ax1.set_ylabel(' ',fontweight='bold')
    ax1.set_title(r'$\epsilon_{3}$',fontsize=15)

    infer_noise_X4={'True':epsilon[:,4].reshape(-1),'full':SCM_inferred_noise['X4_base'].detach().numpy().reshape(-1)}
    sns.kdeplot(infer_noise_X4['True'], ax=ax2, color='red',legend=True,label='True',alpha=0.2,fill=True)
#sns.histplot(infer_noise_X1['partial'], ax=ax2,color='black',legend=True,label='partial model',alpha=0.2,kde=True)
    sns.kdeplot(infer_noise_X4['full'], ax=ax2,color='green',legend=True,label='full model',alpha=1,linewidth=3,linestyle='-')
    ax2.legend()
    ax2.set_ylabel(' ',fontweight='bold')
    ax2.set_title(r'$\epsilon_{4}$',fontsize=15)


    infer_noise_X5={'True':epsilon[:,5].reshape(-1),'full':SCM_inferred_noise['X5_base'].detach().numpy().reshape(-1)}
    sns.kdeplot(infer_noise_X5['True'], ax=ax3, color='red',legend=True,label='True',alpha=0.2,fill=True)
#sns.histplot(infer_noise_X2['partial'], ax=ax3,color='black',legend=True,label='partial model',alpha=0.2,kde=True)
    sns.kdeplot(infer_noise_X5['full'], ax=ax3,color='green',legend=True,label='full model',alpha=1,linewidth=3,linestyle='-')
    ax3.legend()
    ax3.set_ylabel(' ',fontweight='bold')
    ax3.set_title(r'$\epsilon_{5}$',fontsize=15)

    infer_noise_X6={'True':epsilon[:,6].reshape(-1),'full':SCM_inferred_noise['X6_base'].detach().numpy().reshape(-1)}
    sns.kdeplot(infer_noise_X6['True'], ax=ax4, color='red',legend=True,label='True',alpha=0.2,fill=True)
#sns.histplot(infer_noise_X2['partial'], ax=ax3,color='black',legend=True,label='partial model',alpha=0.2,kde=True)
    sns.kdeplot(infer_noise_X6['full'], ax=ax4,color='green',legend=True,label='full model',alpha=1,linewidth=3,linestyle='-')
    ax4.legend()
    ax4.set_ylabel(' ',fontweight='bold')
    ax4.set_title(r'$\epsilon_{6}$',fontsize=15)


    fig.savefig(path+'additional_inferred_noises.pdf',pad_inches=0.1,bbox_inches='tight',format='pdf',dpi=900)
    

def plot_Y_cf(Y_cf=None,SCM_partial_counterfactuals=None,j=None,SCM_counterfactuals=None,path=None):    
    data={'Y':Y_cf,'partial':SCM_partial_counterfactuals['Y'].detach().numpy().reshape(-1),'full':SCM_counterfactuals['Y'].detach().numpy().reshape(-1)}
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data['Y'], ax=ax, color='red', fill=True,bw_adjust=2,legend=True,label='True',alpha=0.1,linewidth=1)
    sns.kdeplot(data['full'], ax=ax, color='green',linestyle='-' ,fill=True,bw_adjust=2,legend=True,label='full model',linewidth=1)
    sns.kdeplot(data['partial'], ax=ax,color='black',linestyle='--' ,fill=True,bw_adjust=2,legend=True,label='partial model',linewidth=1)
    plt.legend()
    plt.ylabel('Density',fontweight='bold')
    plt.title(r'$P(Y_{X_{5}\leftarrow %.2f}|\mathbf{X}=\mathbf{x}^{obs},Y=y^{obs})$'%j,fontsize=15)
    fig.savefig(path+'toy_model.pdf',format='pdf',pad_inches=0.1,bbox_inches='tight',dpi=600)

    
