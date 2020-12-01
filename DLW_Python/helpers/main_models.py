# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:45:34 2020

@author: zhangdongcheng
"""
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F

#import torch.optim as optim
from torchkit import optim
from torch.autograd import Variable
from torchkit import nn as nn_, flows, utils
from sklearn.model_selection import train_test_split

class MAF(object):
    
    def __init__(self, args, p):

        self.args = args        
        self.__dict__.update(args.__dict__)
        self.p = p
        
        dim = p
        dimc = 1
        flowtype = args.flow_type
        
        dimh = args.dimh
        num_flow_layers = args.num_flow_layers
        
        num_ds_dim = args.num_ds_dim
        num_ds_layers = args.num_ds_layers
        fixed_order = args.fixed_order
                 
        act = nn.ELU()
        if flowtype == 'affine':
            flow = flows.IAF
        elif flowtype == 'dsf':
            flow = lambda **kwargs:flows.IAF_DSF(num_ds_dim=num_ds_dim,
                                                 num_ds_layers=num_ds_layers,
                                                 **kwargs)
        elif flowtype == 'ddsf':
            flow = lambda **kwargs:flows.IAF_DDSF(num_ds_dim=num_ds_dim,
                                                  num_ds_layers=num_ds_layers,
                                                  **kwargs)
        
        
        sequels = [nn_.SequentialFlow(
            flow(dim=dim,
                 hid_dim=dimh,
                 context_dim=dimc,
                 num_layers=args.num_hid_layers+1,
                 activation=act,
                 fixed_order=fixed_order),
            flows.FlipFlow(1)) for i in range(num_flow_layers)] + \
                  [flows.LinearFlow(dim, dimc),]
                
                
        self.flow = nn.Sequential(
                *sequels)
        
        
        
        if self.cuda:
            self.flow = self.flow.cuda()
        
        
        
        
    def density(self, spl):
        n = spl.size(0)
        context = Variable(torch.FloatTensor(n, 1).zero_()) 
        lgd = Variable(torch.FloatTensor(n).zero_())
        zeros = Variable(torch.FloatTensor(n, self.p).zero_())
        if self.cuda:
            context = context.cuda()
            lgd = lgd.cuda()
            zeros = zeros.cuda()
            
        z, logdet, _ = self.flow((spl, lgd, context))
#        losses = - utils.log_normal(z, zeros, zeros+1.0).sum(1) - logdet
        losses = - utils.log_normal(z, zeros, zeros).sum(1) - logdet
        return - losses

    def loss(self, x):
        return - self.density(x)
        
    def state_dict(self):
        return self.flow.state_dict()

    def load_state_dict(self, states):
        self.flow.load_state_dict(states)
         
    def clip_grad_norm(self):
        nn.utils.clip_grad_norm_(self.flow.parameters(),
                                self.clip)



class model(object):
        
    def __init__(self, args, filename):
        
        self.__dict__.update(args.__dict__)

        self.filename = filename
        self.args = args 
        self.patience = args.patience
        
        p = args.d
        self.maf = MAF(args, p)
        
        # optim
        amsgrad = bool(args.amsgrad)
        polyak = args.polyak
        self.optim = optim.Adam(self.maf.flow.parameters(),
                                lr=args.lr, 
                                betas=(args.beta1, args.beta2),
                                amsgrad=amsgrad,
                                polyak=polyak)
        
        
        # initialize checkpoint
        self.checkpoint = dict()
        self.checkpoint['best_val'] = float('inf')
        self.checkpoint['best_val_epoch'] = 0
        self.checkpoint['e'] = 0
    
     
    def train(self, epoch, data_train_loader, data_val_loader):

        optim = self.optim
        t = 0 
        
        LOSSES = 0
        counter = 0
        
        #for e in range(epoch):
        while self.checkpoint['e'] < epoch:
            for x in data_train_loader:
                optim.zero_grad()
                x = Variable(x[0])
                if self.cuda:
                    x = x.cuda()
                    
                losses = self.maf.loss(x)
                
                loss = losses.mean()
                
                LOSSES += losses.sum().data.cpu().numpy()
                counter += losses.size(0)

                loss.backward()
                self.maf.clip_grad_norm()
                optim.step()
                t += 1
                
                

            if self.checkpoint['e']%1 == 0:      
                optim.swap()
                loss_val = self.evaluate(data_val_loader)
                print ('Epoch: [%4d/%4d] train <= %.2f ' \
                      'valid: %.3f ' % \
                (self.checkpoint['e']+1, epoch, LOSSES/float(counter), 
                 loss_val))
                if loss_val < self.checkpoint['best_val']:
                    print(' [^] Best validation loss [^] ... [saving]')
                    self.save(self.save_dir+'/'+self.filename+'_best')
                    self.checkpoint['best_val'] = loss_val
                    self.checkpoint['best_val_epoch'] = self.checkpoint['e']+1
                    
                LOSSES = 0
                counter = 0
                optim.swap()
                
            self.checkpoint['e'] += 1
            if (self.checkpoint['e'])%5 == 0:
                self.save(self.save_dir+'/'+self.filename+'_last')
            
            if self.impatient():
                print ('Terminating due to impatience ... \n')
                break 
            
        # loading best valid model (early stopping)
        self.load(self.save_dir+'/'+self.filename+'_best')

    def impatient(self):
        current_epoch = self.checkpoint['e']
        bestv_epoch = self.checkpoint['best_val_epoch']
        return current_epoch - bestv_epoch > self.patience
        
        
    def evaluate(self, dataloader):
        LOSSES = 0 
        c = 0
        for x in dataloader:
            x = Variable(x[0])
            if self.cuda:
                x = x.cuda()
                
            losses = self.maf.loss(x).data.cpu().numpy()
            LOSSES += losses.sum()
            c += losses.shape[0]
        return LOSSES / float(c)


    def save(self, fn):        
        torch.save(self.maf.state_dict(), fn+'_model.pt')
        torch.save(self.optim.state_dict(), fn+'_optim.pt')
        with open(fn+'_args.txt','w') as out:
            out.write(json.dumps(self.args.__dict__,indent=4))
        with open(fn+'_checkpoint.txt','w') as out:     
            out.write(json.dumps(self.checkpoint,indent=4))

    def load(self, fn):
        self.maf.load_state_dict(torch.load(fn+'_model.pt'))
        self.optim.load_state_dict(torch.load(fn+'_optim.pt'))
    
    def resume(self, fn):
        self.load(fn)
        self.checkpoint.update(
            json.loads(open(fn+'_checkpoint.txt','r').read()))
    

       
    