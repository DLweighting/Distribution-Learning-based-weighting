# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:12:04 2020

@author: zhangdongcheng
"""
import numpy as np

import torch
#import matplotlib.pyplot as plt
from torch.distributions.beta import Beta
from torch.distributions import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli

#from torch.distributions.gamma import Gamma
#from torch.distributions import Uniform
  

def generate_data(simu_class, n, dim, sc, beta1, beta2):
    num = np.random.multinomial(n, [1/3, 1/3, 1/3])
    
    x = torch.tensor([])
    for i in range(dim):

        x_dis1 = MultivariateNormal(torch.zeros(1), torch.eye(1))
        x1 = x_dis1.sample([num[0],])
        
        x_dis2 = MultivariateNormal(-3*torch.ones(1), torch.eye(1))
        x2 = x_dis2.sample([num[1],])
    
        x_dis3 = MultivariateNormal(3*torch.ones(1), torch.eye(1))
        x3 = x_dis3.sample([num[2],])
    
        xx = torch.cat((x1,x2,x3))
        xx = xx[torch.randperm(xx.size(0)),:]
        
        x = torch.cat((x, xx), dim=1)

    for i in range(dim):
        x[:,i] = (x[:,i] - x[:,i].mean())/x[:,i].std()
    
    linear = 0
    if simu_class == "simu1":
        for i in range(0,dim):
            linear = linear + (x[:,i]*sc).unsqueeze(1)
    if simu_class == "simu2":
        for i in range(0,dim):       
            linear = linear + ((x[:,i]**2-1)*sc).unsqueeze(1)
        for i in range(0,dim-1):
            for j in range(i+1,dim):
                linear = linear + (x[:,i]*x[:,j]*sc).unsqueeze(1)
    if simu_class == "simu3":
        for i in range(0,dim):       
            linear = linear + ((torch.log(1+x[:,i]**2)-0.5) * sc).unsqueeze(1)
        for i in range(0,dim-1):
            for j in range(i+1,dim):
                linear = linear + (x[:,i]*x[:,j]*sc).unsqueeze(1)
    
    prob_w = 1/(1+torch.exp(0.5*linear)) # prob_w.shape=[n,1]
    w = torch.tensor([Bernoulli(prob_w[i]).sample() for i in range(n)])
    
    noise_dist = MultivariateNormal(torch.zeros(1), torch.eye(1))# create the data noise, which is the standard Gaussian
    noise0 = noise_dist.sample([n,])
    noise1 = noise_dist.sample([n,])


    y0 = 0
    if simu_class == "simu1":
        y0 = torch.mm(x, beta1)

    if simu_class == "simu2":
        y0 = torch.mm(x**2, beta1)
        index = 0
        for i in range(0,dim-1):
            for j in range(i+1,dim):
                y0 = y0 + (x[:,i] * x[:,j] * beta2[index]).unsqueeze(1)
                index = index + 1   

    if simu_class == "simu3":
        y0 = torch.mm(torch.log(x**2 +1),beta1)
        index = 0
        for i in range(0,dim-1):
            for j in range(i+1,dim):
                y0 = y0 + (2*torch.sin(x[:,i]*x[:,j]) * beta2[index]).unsqueeze(1)
                index = index + 1    
    
    y1 = y0 + 1 + noise1
    y0 = y0 + noise0
    
    y = torch.tensor([y0[i] if w[i]==0 else y1[i] for i in range(n)])
    y  = y.unsqueeze(1)

    return (x, w, prob_w, y, y0 ,y1) 
    
