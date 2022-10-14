from ast import arg
import os
import time
import torch
import shutil
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing as mp
import pyro

import models
from utils import mkdir, milestone_step
from loaders import get_features_dataset


def save_checkpoint(state, is_best, filepath):
    mkdir(filepath)
    torch.save(state, os.path.join(filepath, 'flow_ckpt.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'flow_ckpt.pth.tar'), os.path.join(filepath, 'flow_best.pth.tar'))
        #print('best is saved')
#---------------------------------------------------------------------------------------------------------------


def train(args, model, optimizer, train_loader, epoch):
    avg_loss = 0.
    for batch_idx, (sex,age, amount, duration) in enumerate(train_loader):
        if args.cuda:
            amount, age, sex, duration = amount.reshape([-1,1]).cuda(), age.reshape([-1,1]).cuda(), sex.reshape([-1,1]).cuda(), duration.reshape([-1,1]).cuda(), #risk.reshape(64,1).cuda()
        optimizer.zero_grad()
        #print('amount',amount,amount.shape)
        #print('sex',sex,sex.shape)
        #print('age',age,age.shape)
        #print('Duration-----------:',duration)
        log_p = model(sex, age ,amount, duration)
        loss = -torch.mean(log_p['sex']+log_p['age']+log_p['amount']+log_p['duration']) #log_p['risk'])
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
        model.clear()
        #if epoch % args.log_interval ==0 and batch_idx % args.log_interval == 0:
           # print('Train Epoch: {} [{}/{} ({:.1f}%)]\t| -LogProb Sex: {:.6f}\tAge: {:.6f} \tAmount: {:.6f}\tDuration: {:.6f} \tTotal: {:.6f}'.format(epoch, batch_idx * len(duration),
            #    len(train_loader.dataset), 100. * batch_idx / len(train_loader),
            #    torch.mean(log_p['sex']).item(), torch.mean(log_p['age']).item(),
            #    torch.mean(log_p['amount']).item(), torch.mean(log_p['duration']).item(), -loss.item()))

            

def test(args, model, test_loader):
    test_loss = 0.
    model.eval()
    associative_power ={'sex':0,'age':0,'amount':0,'duration':0}
    for sex,age, amount, duration in test_loader:
        with torch.no_grad():
            if args.cuda:
                amount, age, sex, duration= amount.reshape([-1,1]).cuda(), age.reshape([-1,1]).cuda(), sex.reshape([-1,1]).cuda(), duration.reshape([-1,1]).cuda() #risk.cuda()
            log_p = model(sex, age, amount, duration)
            test_loss += torch.mean(log_p['sex']+log_p['age']+log_p['amount']+log_p['duration'])
                                    #log_p['risk']).item()
            for k,v in log_p.items():
                associative_power[k]+=torch.mean(v)

    test_loss /= len(test_loader)
    associative_power=dict(associative_power) 
    associative_power={k: v/len(test_loader) for k,v in associative_power.items()}
    #print('\nTest set: Average  LogProb: {:.6f}\n'.format(test_loss))
    return test_loss, associative_power





#--------------------------------------------------------------------------------------------------------------
def main(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    #print(kwargs)
    dataset_train, dataset_test = get_features_dataset(
        filename=args.data_filename, dim=args.dim, random_seed=args.data_seed)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    print('model:',args.arch,'flow type:',args.flow_type,'flow order:',args.flow_order)
    print(args)
    model = models.__dict__[args.arch](
    flow_dict=dataset_train.flow_dict, flow_type=args.flow_type, order=args.flow_order)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0)

    best_loss = -100.
    start_time = time.time()
    for epoch in range(args.epochs):
        if not args.lr_annealing:
            milestone_step(args, optimizer, epoch)
        else:
            scheduler.step()
        train(args, model, optimizer, train_loader, epoch)
        loss, ass_power = test(args, model, test_loader)
        is_best = loss > best_loss
        best_loss = max(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'associative power':ass_power
        }, is_best, filepath=args.save)

    print('==> Best LogProb: {:.6f}, Time: {:.2f} min\n'.format(best_loss, (time.time()-start_time)/60.))
    del model
    with torch.cuda.device('cuda:' + args.gpu_id):
        torch.cuda.empty_cache()
    print('Done...\n\n\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Flow SCM')
    parser.add_argument('--data-filename', default='features_data.csv', type=str, metavar='PATH',
                    help='dataset csv file name')
    parser.add_argument('--dim', type=int, default=1,
                    help='dimension of the data features (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='learning rate (default: 3e-4)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default='0',
                    help='gpu id')
    parser.add_argument('--data-seed', type=int, default=69, metavar='S',
                    help='dataset seed (default: 69)')
    parser.add_argument('--seed', type=int, default=9, metavar='S',
                    help='random seed (default: 9)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
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
    parser.add_argument('--lr-annealing', action='store_true', default=False,
                    help='annealing learning rate (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.set_device('cuda:' + args.gpu_id)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        pyro.set_rng_seed(args.seed)

    mkdir(args.save)
    args.save = os.path.join(args.save, args.arch + '_flowtype_' + args.flow_type
        + '_floworder_' + args.flow_order)
    mkdir(args.save)
    main(args)
    
#Saptarshi Saha