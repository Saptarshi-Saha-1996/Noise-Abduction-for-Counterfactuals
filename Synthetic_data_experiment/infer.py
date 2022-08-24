import os
import torch
#save='./logs_2'

def do_inference(model_name,intervention,save=None,data=None,print_mode=False):
    model = model_name
    saved_in_path= os.path.join(save,str(model_name.__class__.__name__)) 
    model_path = os.path.join(saved_in_path, 'flow_best.pth.tar')
    checkpoint = torch.load(model_path)
    if print_mode:
        print('\n',model_path)
        print('====>>> Best LogProb: {:.6f}\n'.format(checkpoint['best_loss']))
        print('associative power:',checkpoint['associative power'])
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if print_mode:
        print('intervention------',intervention)

    if 'partial' in model.__class__.__name__:
        Samples=model.partial_sample(n_samples=1000,**data)  
        samples={'X6':Samples[0].detach(),'X5':Samples[1].detach(),'X4':Samples[2].detach(),'X3':Samples[3].detach(),'X2':Samples[4].detach(),'X1':Samples[5].detach(),'Y':Samples[6].detach()}
    else:
        Samples=model.sample(n_samples=1000)  
        samples={'X6':Samples[0].detach(),'X5':Samples[1].detach(),'X4':Samples[2].detach(),'X3':Samples[3].detach(),'X2':Samples[4].detach(),'X1':Samples[5].detach(),'Y':Samples[6].detach()}

    Exogeneous_noise=model.infer_exogeneous(**data)
    Counterfactuals = model.counterfactual(obs=data, intervention=intervention) 
    
    return samples ,Exogeneous_noise, Counterfactuals
