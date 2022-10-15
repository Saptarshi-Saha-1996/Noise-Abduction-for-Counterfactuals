import os
import time
import torch


import shutil
import torch.optim as optim
import torch.nn.functional as F
#import multiprocessing as mp
import numpy as np
from utils import mkdir
#from collections import Counter

from data import get_features_dataset 
from config import *



#NUM_WORKERS=12

def milestone_step( optimizer, epoch):
    if epoch in [epochs*0.5, epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def save_checkpoint(state, is_best, filepath):
    mkdir(filepath)
    torch.save(state, os.path.join(filepath, 'flow_ckpt.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'flow_ckpt.pth.tar'), os.path.join(filepath, 'flow_best.pth.tar'))
        #print('best is saved')
#---------------------------------------------------------------------------------------------------------------


def train( model, optimizer, train_loader, epoch, print_mode=True):
    avg_loss = 0.
    for batch_idx, (X6,X5,X4,X3, X2, X1,Y) in enumerate(train_loader):
        X6, X5, X4, X3, X2, X1,Y = X6.reshape([-1,1]), X5.reshape([-1,1]), X4.reshape([-1,1]), X3.reshape([-1,1]), X2.reshape([-1,1]), X1.reshape([-1,1]), Y.reshape([-1,1]) 
        optimizer.zero_grad()
        log_p = model(X6, X5, X4, X3 ,X2, X1,Y)
        loss = -torch.mean(log_p['X_6']+log_p['X_5']+log_p['X_4']+log_p['X_3']+log_p['X_2']+log_p['X_1'] +log_p['Y'])
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
        model.clear()

    avg_loss /= -len(train_loader)
    if epoch % log_interval ==0 and print_mode==True:
        print('Epoch:',epoch,'\nTrain set: Average LogProb: {:.6f}\n'.format(avg_loss))
    
            
            

def test(model, test_loader):
    test_loss = 0.
    model.eval()
    associative_power ={'X_6':0,'X_5':0,'X_4':0,'X_3':0,'X_2':0,'X_1':0,'Y':0}
    for X6, X5, X4, X3, X2, X1,Y in test_loader:
        with torch.no_grad():
            X6, X5, X4, X3, X2, X1,Y = X6.reshape([-1,1]), X5.reshape([-1,1]), X4.reshape([-1,1]), X3.reshape([-1,1]), X2.reshape([-1,1]), X1.reshape([-1,1]), Y.reshape([-1,1]) 
            log_p = model(X6, X5, X4, X3 ,X2, X1,Y)
            test_loss += torch.mean(log_p['X_6']+log_p['X_5']+log_p['X_4']+log_p['X_3']+log_p['X_2']+log_p['X_1'] +log_p['Y'])
            for k,v in log_p.items():
                associative_power[k]+=torch.mean(v)
                #print(k,v)
                
                
    #print('length',len(test_loader))            
    test_loss /= len(test_loader)   #len(test_loader.dataset)
    associative_power=dict(associative_power) 
    associative_power={k: v/len(test_loader) for k,v in associative_power.items()}

    return test_loss, associative_power



def main(model_name,filepath,seed=None,data=None,print_mode=True):
    kwargs =  {}
    dataset_train, dataset_test = get_features_dataset(data=data, dim=1,random_seed=seed,sample_size=sample_size)
    np.random.seed(seed)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False, **kwargs)

    model = model_name
 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') 

    best_loss = -500.
    start_time = time.time()
    for epoch in range(epochs):      
        train( model, optimizer, train_loader, epoch,print_mode=print_mode)
        loss,ass_power = test(model, test_loader)
        if epoch%log_interval==0 and print_mode==True:
            print('\nTest set: Average LogProb: {:.6f}\n'.format(loss))
        if not lr_annealing:
            milestone_step(optimizer, epoch)
        #else:
            #scheduler.step() #loss)
        
        is_best = loss > best_loss
        best_loss = max(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'associative power':ass_power
        }, is_best, filepath=filepath)

    time_=(time.time()-start_time)/60.
    
    del model
    
    if print_mode:
        print('==> Best LogProb: {:.6f}, Time: {:.2f} min\n'.format(best_loss,time_ ))
    return time_
    

    
def train_model(model,path,data=None,seed=None,print_mode=True):
    mkdir(path)
    path = os.path.join(path, str(model.__class__.__name__))
    mkdir(path)
    filepath=path
    time=main(model,filepath,seed=seed,data=data,print_mode=print_mode)
    return time
    
    