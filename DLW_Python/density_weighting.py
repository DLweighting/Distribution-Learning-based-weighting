# -*- coding: utf-8 -*-

import numpy as np

import time
import torch
import os
import sys 

import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import importlib


from helpers.main_models import model

from helpers.auxiliary_functions import generate_data
torch.set_num_threads(1)


np.random.seed(2020)
torch.manual_seed(2020)    
# =============================================================================
# main
# =============================================================================


"""parsing and configuration"""
def parse_args():
    desc = "MAF"
    parser = ArgumentParser(description=desc)


    parser.add_argument('--epoch', type=int, default=2000, 
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--amsgrad', type=int, default=0)
    parser.add_argument('--polyak', type=float, default=0.0)
    parser.add_argument('--cuda', type=bool, default=False)
    

    parser.add_argument('--patience', type=int, default=200)

    parser.add_argument('--flow_type', type=str, default='dsf')
    parser.add_argument('--num_flow_layers', type=int, default=6)
    
    parser.add_argument('--dimh', type=int, default=64)
    parser.add_argument('--num_hid_layers', type=int, default=2)
    
    parser.add_argument('--num_ds_dim', type=int, default=8)
    parser.add_argument('--num_ds_layers', type=int, default=1)
    parser.add_argument('--fixed_order', type=bool, default=True,
                        help='Fix the made ordering to be the given order')

    parser.add_argument('--simu_class', type=str, default = "simu1")
    parser.add_argument('--d', type=int, default = 8) # d=8,16
    parser.add_argument('--n', type=int, default = 10000) # n=5000,10000
    parser.add_argument('--sc', type=float, default = 0.8) # sc=0.4,0.8

    parser.add_argument('--loaddata', default = True, action='store_false', help='Bool type') # to be false, pass --loaddata
    
    parser.add_argument('--times', type=int, default = 100)    
    
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args
   

def args2fn(args):
    
    prefix_key_pairs = [
        ('simu_class_', 'simu_class'),
        ('d', 'd'),
        ('n', 'n'),
        ('sc', 'sc'),
        ('e', 'epoch'),
        ('lr', 'lr'),
        ('batch_size', 'batch_size'),
        ('p', 'polyak'),
        ('patience', 'patience'),
        ('f', 'flow_type'),
        ('flowlayer', 'num_flow_layers'),
        ('h', 'dimh'),
        ('hidlayer', 'num_hid_layers'),
        ('dslayer', 'num_ds_layers'),
        ('dsdim','num_ds_dim'),
        ('loaddata_', 'loaddata'),
        ('times_', 'times')
    ]
    
    return '_'.join([p+str(args.__dict__[k]) for p, k in prefix_key_pairs])
    



def data_prepare_weighting(n, d, sc, seed):
    # data generation
    device = torch.device("cpu")
    print("The device is", device)
       
    if args.loaddata:
        data_name = "data_simu/class_"+ simu_class+"_d"+str(d)+"_n"+str(n)+"_sc"+str(sc)+"_data_all_"+str(seed)+".csv"
        # data_name = "realdata_twins/real_twins_data_all_"+str(seed)+".csv"
        
        data = pd.read_csv(data_name)
        
        data_array=data.to_numpy()
        data_torch=torch.from_numpy(data_array)
        data_all =(data_torch[:,0:d].float(), data_torch[:,d].float(), \
         data_torch[:,d+1].unsqueeze(1).float(),data_torch[:,d+2].unsqueeze(1).float(), \

         data_torch[:,d+5].unsqueeze(1).float(), data_torch[:,d+6].unsqueeze(1).float(), \
         data_torch[:,d+7].unsqueeze(1).float(), data_torch[:,d+8].unsqueeze(1).float(), \
         data_torch[:,d+9].unsqueeze(1).float(), \

         data_torch[:,d+10].unsqueeze(1).float(),data_torch[:,d+11].unsqueeze(1).float(),\
         data_torch[:,d+12].unsqueeze(1).float(),data_torch[:,d+13].unsqueeze(1).float(),\

         data_torch[:,d+14].unsqueeze(1).float(),data_torch[:,d+15].unsqueeze(1).float()
         )
    
        data_all = (data_all[0].to(device), data_all[1].to(device), data_all[2].to(device), \
                    data_all[3].to(device), \

                    data_all[4].to(device), data_all[5].to(device), data_all[6].to(device), \
                    data_all[7].to(device), data_all[8].to(device),\

                    data_all[9].to(device), data_all[10].to(device), \
                    data_all[11].to(device), data_all[12].to(device),\

                    data_all[13].to(device), data_all[14].to(device))
    else:
        if d==8:
            beta1 = torch.tensor([0.7599586, 0.8201394, 0.6193205, 0.4586492, 0.5157641,
                0.2891778, 0.5111296, 0.0548527]).unsqueeze(1)
            beta2 = pd.read_csv("beta2_d8.txt", header=None)
            beta2 = torch.tensor(beta2.to_numpy().squeeze(0))

        if d==16:
            beta1 = torch.tensor([0.50013665, 0.93067848, 0.85806673, 0.40690777, 0.51404529, 0.66080241,
             0.61336544, 0.65048289, 0.40307454, 0.97905294, 0.01737927, 0.31150095,
             0.15645899, 0.17930328, 0.55058790, 0.04315068]).unsqueeze(1)
            beta2 = pd.read_csv("beta2_d16.txt", header=None)
            beta2 = torch.tensor(beta2.to_numpy().squeeze(0))

        data_all = generate_data(simu_class, n, d, sc, beta1, beta2)

    y1_index=(data_all[1]==1).nonzero().squeeze() # get the index of w=1
    y0_index=(data_all[1]==0).nonzero().squeeze()
    data_1=(data_all[0][y1_index]) # get the data of w=1
    data_0=(data_all[0][y0_index])
    print("data_1_all number is",len(y1_index))
    print("data_0_all number is",len(y0_index))
    
    (input_train, input_val) = train_test_split(data_0.cpu().numpy(), test_size=0.20, random_state=4)    
    data_0_train=(torch.from_numpy(input_train).to(device))    
    data_0_val=(torch.from_numpy(input_val).to(device))
    
    (input_train, input_val) = train_test_split(data_1.cpu().numpy(), test_size=0.20, random_state=4)    
    data_1_train=(torch.from_numpy(input_train).to(device))    
    data_1_val=(torch.from_numpy(input_val).to(device)) 
    
    del input_train, input_val


    data_0_train_dataset = Data.TensorDataset(data_0_train)
    data_0_train_loader  = Data.DataLoader(data_0_train_dataset, batch_size = batch, shuffle=True)
    print('Number of train_0: ', len(data_0_train_dataset))

    data_0_val_dataset = Data.TensorDataset(data_0_val)
    data_0_val_loader  = Data.DataLoader(data_0_val_dataset, batch_size = batch, shuffle=True)
    print('Number of val_0: ', len(data_0_val_dataset))    
    
    data_1_train_dataset = Data.TensorDataset(data_1_train)
    data_1_train_loader  = Data.DataLoader(data_1_train_dataset, batch_size = batch, shuffle=True)
    print('Number of train_1: ', len(data_1_train_dataset))

    data_1_val_dataset = Data.TensorDataset(data_1_val)
    data_1_val_loader  = Data.DataLoader(data_1_val_dataset, batch_size = batch, shuffle=True)
    print('Number of val_1: ', len(data_1_val_dataset)) 
    
    return (data_all, data_0_train_loader, data_0_val_loader, data_1_train_loader, data_1_val_loader)
    
    

"""main"""
def main(seed):
    (data_all, data_0_train_loader, data_0_val_loader, data_1_train_loader, data_1_val_loader) = data_prepare_weighting(n, d, sc, seed)
    
    #fn = str(time.time()).replace('.','')
    fn = args2fn(args)
    print (args)
    print ('\nfilename: ', fn)
    print(" [*] Building model!")

    ############## for data_0, train one model
    mdl_0 = model(args, fn+"_y0")
    # launch the graph in a session
    print(" [*] Control_Training started!")
#    mdl_0.train(1, data_0_train_loader, data_0_val_loader)
    mdl_0.train(args.epoch, data_0_train_loader, data_0_val_loader)
    print(" [*] Control_Training finished!")

    print (" [**] Valid: %.4f" % mdl_0.evaluate(data_0_val_loader))
    print (" [**] Valid best epoch is: %d" % mdl_0.checkpoint['best_val_epoch'])
    print(" [*] Validation_0 finished!")

    ############## for data_1, train another model
    mdl_1 = model(args, fn+"_y1")
    print(" [*] Treat_Training started!")
    mdl_1.train(args.epoch, data_1_train_loader, data_1_val_loader)
    print(" [*] Treat_Training finished!") 
        
    print (" [**] Valid: %.4f" % mdl_1.evaluate(data_1_val_loader))
    print (" [**] Valid best epoch is: %d" % mdl_1.checkpoint['best_val_epoch'])
    print(" [*] Validation_1 finished!")
    
    
    
    with torch.no_grad():
        
        y1_index=(data_all[1]==1).nonzero().squeeze() # get the index of w=1
        y0_index=(data_all[1]==0).nonzero().squeeze()
    
        prob_0 = torch.exp(mdl_0.maf.density(data_all[0]))
        prob_1 = torch.exp(mdl_1.maf.density(data_all[0]))

        # 
        if args.loaddata == False:
            data_1=(data_all[0][y1_index], data_all[1][y1_index], data_all[2][y1_index].squeeze(1), 
            data_all[3][y1_index].squeeze(1)) # get the data of w=1

            data_0=(data_all[0][y0_index], data_all[1][y0_index], data_all[2][y0_index].squeeze(1), 
            data_all[3][y0_index].squeeze(1)) # get the data of w=0

            # base error        
            ATT = (data_1[3].mean() - data_0[3].mean()).item()
            print("The base average treatment effect for treated is", ATT)              
            ATT_base.append(ATT)
        
        # Implementing weighting for ATT
            weights = prob_1[y0_index]/prob_0[y0_index]
            ATT = (data_1[3].mean() - (data_0[3] * weights).mean()).item()
            print("The DLW ATT is", ATT)
            ATT_DLW.append(ATT)
            
            weight = weights/torch.sum(weights)
            ATT = (data_1[3].mean() - (data_0[3] * weight).sum()).item()
            print("The adjusted DLW ATT is", ATT)
            ATT_DLW_adjust.append(ATT)
            
            # Using true p-score weighting for ATT
            ATT = data_1[3].mean() - (data_0[3] * data_0[2]/(1-data_0[2])).sum()/len(y1_index)
            ATT = ATT.item()
            print("The true weighting ATT is", ATT)
            ATT_weighting.append(ATT)
            
            ATT = data_1[3].mean() - (data_0[3] * data_0[2]/(1-data_0[2])).sum()/(data_0[2]/(1-data_0[2])).sum()
            ATT = ATT.item()
            print("The true adjusted weighting ATT is", ATT)
            ATT_weighting_adjust.append(ATT)

            return


        # part the data into treatment and control groups
        data_1=(data_all[0][y1_index], data_all[1][y1_index], data_all[2][y1_index].squeeze(1), 
        data_all[3][y1_index].squeeze(1), \

        data_all[4][y1_index].squeeze(1), data_all[5][y1_index].squeeze(1),\
        data_all[6][y1_index].squeeze(1), data_all[7][y1_index].squeeze(1),\
        data_all[8][y1_index].squeeze(1), \

        data_all[9][y1_index].squeeze(1), data_all[10][y1_index].squeeze(1), \
        data_all[11][y1_index].squeeze(1), data_all[12][y1_index].squeeze(1), \

        data_all[13][y1_index].squeeze(1), data_all[14][y1_index].squeeze(1)) # get the data of w=1


        data_0=(data_all[0][y0_index], data_all[1][y0_index], data_all[2][y0_index].squeeze(1), 
        data_all[3][y0_index].squeeze(1), \

        data_all[4][y0_index].squeeze(1), data_all[5][y0_index].squeeze(1),\
        data_all[6][y0_index].squeeze(1), data_all[7][y0_index].squeeze(1),\
        data_all[8][y0_index].squeeze(1), \

        data_all[9][y0_index].squeeze(1), data_all[10][y0_index].squeeze(1), \
        data_all[11][y0_index].squeeze(1), data_all[12][y0_index].squeeze(1), \

        data_all[13][y0_index].squeeze(1), data_all[14][y0_index].squeeze(1)) # get the data of w=0


        # base error        
        ATT = (data_1[3].mean() - data_0[3].mean()).item()
        print("The base average treatment effect for treated is", ATT)              
        ATT_base.append(ATT)
        
        ##################################################################### 
        # weighting and matching
        
        ################################# For ATT       
        
        # Implementing weighting for ATT
        weights = prob_1[y0_index]/prob_0[y0_index]
        ATT = (data_1[3].mean() - (data_0[3] * weights).mean()).item()
        print("The DLW ATT is", ATT)
        ATT_DLW.append(ATT)
        
        weight = weights/torch.sum(weights)
        ATT = (data_1[3].mean() - (data_0[3] * weight).sum()).item()
        print("The adjusted DLW ATT is", ATT)
        ATT_DLW_adjust.append(ATT)
        
        # Using true p-score weighting for ATT
        ATT = data_1[3].mean() - (data_0[3] * data_0[2]/(1-data_0[2])).sum()/len(y1_index)
        ATT = ATT.item()
        print("The true weighting ATT is", ATT)
        ATT_weighting.append(ATT)
        
        ATT = data_1[3].mean() - (data_0[3] * data_0[2]/(1-data_0[2])).sum()/(data_0[2]/(1-data_0[2])).sum()
        ATT = ATT.item()
        print("The true adjusted weighting ATT is", ATT)
        ATT_weighting_adjust.append(ATT)

        ################## baseline weighting methods
        #### statistical based
        # using weights_ebal
        ATT = data_1[3].mean() - (data_0[3] * data_0[4]).sum()/(data_0[4]).sum()
        ATT = ATT.item()
        print("The weights_ebal ATT is", ATT)
        ATT_ebal.append(ATT)        

        # using weights_cbps
        ATT = data_1[3].mean() - (data_0[3] * data_0[5]).sum()/(data_0[5]).sum()
        ATT = ATT.item()
        print("The weights_cbps ATT is", ATT)
        ATT_cbps.append(ATT)   

        # using weights_opt
        ATT = data_1[3].mean() - (data_0[3] * data_0[6]).sum()/(data_0[6]).sum()
        ATT = ATT.item()
        print("The weights_opt ATT is", ATT)
        ATT_opt.append(ATT)  

        # using weights_ebcw
        ATT = data_1[3].mean() - (data_0[3] * data_0[7]).sum()/(data_0[7]).sum()
        ATT = ATT.item()
        print("The weights_ebcw ATT is", ATT)
        ATT_ebcw.append(ATT)  

        # using weights_ps
        ATT = data_1[3].mean() - (data_0[3] * data_0[8]).sum()/(data_0[8]).sum()
        ATT = ATT.item()
        print("The weights_ps ATT is", ATT)
        ATT_ps.append(ATT)  

        #### machine_learing based
        # using rf_weights
        ATT = data_1[3].mean() - (data_0[3] * data_0[9]).sum()/(data_0[9]).sum()
        ATT = ATT.item()
        print("The rf_weights ATT is", ATT)
        ATT_rf_weights.append(ATT)  

        # using treebag_weights
        ATT = data_1[3].mean() - (data_0[3] * data_0[10]).sum()/(data_0[10]).sum()
        ATT = ATT.item()
        print("The treebag_weights ATT is", ATT)
        ATT_treebag_weights.append(ATT) 

        # using xgboost_weights
        ATT = data_1[3].mean() - (data_0[3] * data_0[11]).sum()/(data_0[11]).sum()
        ATT = ATT.item()
        print("The xgboost_weights ATT is", ATT)
        ATT_xgboost_weights.append(ATT) 

        # using lasso_weights
        ATT = data_1[3].mean() - (data_0[3] * data_0[12]).sum()/(data_0[12]).sum()
        ATT = ATT.item()
        print("The lasso_weights ATT is", ATT)
        ATT_lasso_weights.append(ATT)

    #####################################################################
    with torch.no_grad():# double robust, for BART
                
        ################################# For ATT
        # based on outcome model for ATT
        mu_0 = data_all[13].squeeze(1)
            
        ATT = (data_1[3].mean() - mu_0[y1_index]).mean().item()
        print("ATT based on outcome_bart is", ATT)
        ATT_outcome_bart.append(ATT)
          
        # Double robust for ATT
    #    v0 = (data_0[3] * weights).mean() + (mu_0.sum()- \
    #          (mu_0[y0_index]/(1-prob_treat_0)).sum())/len(y1_index)
        v0 = ((data_0[3] - mu_0[y0_index])* weights).mean() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The DR_bart ATT is", ATT)
        ATT_DR_bart.append(ATT)
        
        # Double robust adjust for ATT
        v0 = ((data_0[3] - mu_0[y0_index])* weight).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The adjusted DR_bart ATT is", ATT)
        ATT_DR_adjust_bart.append(ATT)
        
        # Using true p-score weighting for double robust ATT
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[2]/(1-data_0[2])).sum()/len(y1_index) +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The true weighting DR_bart ATT is", ATT)
        ATT_DR_weighting_bart.append(ATT)
    
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[2]/(1-data_0[2])).sum()/(data_0[2]/(1-data_0[2])).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The true adjusted weighting DR_bart ATT is", ATT)
        ATT_DR_weighting_adjust_bart.append(ATT)    
    

        ############# baseline weighting method for double robust ATT
        # statistical based methods
        # using weights_ebal
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[4]).sum()/(data_0[4]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_ebal DR_bart ATT is", ATT)
        ATT_DR_ebal_bart.append(ATT)

        # using weights_cbps
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[5]).sum()/(data_0[5]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_cbps DR_bart ATT is", ATT)
        ATT_DR_cbps_bart.append(ATT)    

        # using weights_opt
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[6]).sum()/(data_0[6]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_opt DR_bart ATT is", ATT)
        ATT_DR_opt_bart.append(ATT)   

        # using weights_ebcw
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[7]).sum()/(data_0[7]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_ebcw DR_bart ATT is", ATT)
        ATT_DR_ebcw_bart.append(ATT)  

        # using weights_ps
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[8]).sum()/(data_0[8]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_ps DR_bart ATT is", ATT)
        ATT_DR_ps_bart.append(ATT)  

        ##### machine_learning based methods
        # using weights_rf
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[9]).sum()/(data_0[9]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_rf DR_bart ATT is", ATT)
        ATT_DR_rf_bart.append(ATT)  

        # using weights_treebag
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[10]).sum()/(data_0[10]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_treebag DR_bart ATT is", ATT)
        ATT_DR_treebag_bart.append(ATT)  

        # using weights_xgboost
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[11]).sum()/(data_0[11]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_xgboost DR_bart ATT is", ATT)
        ATT_DR_xgboost_bart.append(ATT)  

        # using weights_lasso
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[12]).sum()/(data_0[12]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_lasso DR_bart ATT is", ATT)
        ATT_DR_lasso_bart.append(ATT)  

    #####################################################################
    with torch.no_grad():# double robust, for random forest
                
        ################################# For ATT
        # based on outcome model for ATT
        mu_0 = data_all[14].squeeze(1)
            
        ATT = (data_1[3].mean() - mu_0[y1_index]).mean().item()
        print("ATT based on outcome_rf is", ATT)
        ATT_outcome_ranf.append(ATT)
          
        # Double robust for ATT
    #    v0 = (data_0[3] * weights).mean() + (mu_0.sum()- \
    #          (mu_0[y0_index]/(1-prob_treat_0)).sum())/len(y1_index)
        v0 = ((data_0[3] - mu_0[y0_index])* weights).mean() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The DR_ranf ATT is", ATT)
        ATT_DR_ranf.append(ATT)
        
        # Double robust adjust for ATT
        v0 = ((data_0[3] - mu_0[y0_index])* weight).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The adjusted DR_ranf ATT is", ATT)
        ATT_DR_adjust_ranf.append(ATT)
        
        # Using true p-score weighting for double robust ATT
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[2]/(1-data_0[2])).sum()/len(y1_index) +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The true weighting DR_ranf ATT is", ATT)
        ATT_DR_weighting_ranf.append(ATT)
    
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[2]/(1-data_0[2])).sum()/(data_0[2]/(1-data_0[2])).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The true adjusted weighting DR_ranf ATT is", ATT)
        ATT_DR_weighting_adjust_ranf.append(ATT)    
    

        ############# baseline weighting method for double robust ATT
        # statistical based methods
        # using weights_ebal
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[4]).sum()/(data_0[4]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_ebal DR_ranf ATT is", ATT)
        ATT_DR_ebal_ranf.append(ATT)

        # using weights_cbps
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[5]).sum()/(data_0[5]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_cbps DR_ranf ATT is", ATT)
        ATT_DR_cbps_ranf.append(ATT)    

        # using weights_opt
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[6]).sum()/(data_0[6]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_opt DR_ranf ATT is", ATT)
        ATT_DR_opt_ranf.append(ATT)   

        # using weights_ebcw
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[7]).sum()/(data_0[7]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_ebcw DR_ranf ATT is", ATT)
        ATT_DR_ebcw_ranf.append(ATT)  

        # using weights_ps
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[8]).sum()/(data_0[8]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_ps DR_ranf ATT is", ATT)
        ATT_DR_ps_ranf.append(ATT)  

        ##### machine_learning based methods
        # using weights_ranf
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[9]).sum()/(data_0[9]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_rf DR_ranf ATT is", ATT)
        ATT_DR_rf_ranf.append(ATT)  

        # using weights_treebag
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[10]).sum()/(data_0[10]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_treebag DR_ranf ATT is", ATT)
        ATT_DR_treebag_ranf.append(ATT)  

        # using weights_xgboost
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[11]).sum()/(data_0[11]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_xgboost DR_ranf ATT is", ATT)
        ATT_DR_xgboost_ranf.append(ATT)  

        # using weights_lasso
        v0 = ((data_0[3] - mu_0[y0_index])* data_0[12]).sum()/(data_0[12]).sum() +  mu_0[y1_index].mean()
        ATT = (data_1[3].mean() - v0).item()
        print("The weights_lasso DR_ranf ATT is", ATT)
        ATT_DR_lasso_ranf.append(ATT)  
        

if __name__ == '__main__':
    
    args = parse_args()
    
    times = args.times
    d = args.d
    n = args.n
    sc = args.sc
    simu_class = args.simu_class

    batch = args.batch_size

    ATT_base = []
    # matching and weighting
    ATT_DLW = []
    ATT_DLW_adjust = []
    ATT_weighting = []
    ATT_weighting_adjust = []

    ATT_ebal = []
    ATT_cbps = []
    ATT_opt = []
    ATT_ebcw = []
    ATT_ps = []

    ATT_rf_weights = []  
    ATT_treebag_weights = []  
    ATT_xgboost_weights = []  
    ATT_lasso_weights = []  

    # double robust, for BART
    ATT_outcome_bart = []
    ATT_DR_bart = [] 
    ATT_DR_adjust_bart = []
    ATT_DR_weighting_bart = []
    ATT_DR_weighting_adjust_bart = []
    
    ATT_DR_ebal_bart = []
    ATT_DR_cbps_bart = []
    ATT_DR_opt_bart = []
    ATT_DR_ebcw_bart = []
    ATT_DR_ps_bart = []

    ATT_DR_rf_bart = []
    ATT_DR_treebag_bart = []
    ATT_DR_xgboost_bart = []
    ATT_DR_lasso_bart = []

    # double robust, for random forest
    ATT_outcome_ranf = []
    ATT_DR_ranf = [] 
    ATT_DR_adjust_ranf = []
    ATT_DR_weighting_ranf = []
    ATT_DR_weighting_adjust_ranf = []
    
    ATT_DR_ebal_ranf = []
    ATT_DR_cbps_ranf = []
    ATT_DR_opt_ranf = []
    ATT_DR_ebcw_ranf = []
    ATT_DR_ps_ranf = []

    ATT_DR_rf_ranf = []
    ATT_DR_treebag_ranf = []
    ATT_DR_xgboost_ranf = []
    ATT_DR_lasso_ranf = []

    
    start=time.time()
    for seed in range(1, 1 + times):
        main(seed)
    
    w=1

    result_list = ["ATT_base", "ATT_DLW", "ATT_DLW_adjust" , "ATT_weighting",\
                   "ATT_weighting_adjust", \

    "ATT_ebal", "ATT_cbps", "ATT_opt", "ATT_ebcw", "ATT_ps",\
    "ATT_rf_weights",  "ATT_treebag_weights", "ATT_xgboost_weights", 
    "ATT_lasso_weights",\

    "ATT_outcome_bart", "ATT_DR_bart", "ATT_DR_adjust_bart", "ATT_DR_weighting_bart",\
     "ATT_DR_weighting_adjust_bart", \

    "ATT_DR_ebal_bart", "ATT_DR_cbps_bart", "ATT_DR_opt_bart", "ATT_DR_ebcw_bart", \
    "ATT_DR_ps_bart",\
    "ATT_DR_rf_bart",  "ATT_DR_treebag_bart", "ATT_DR_xgboost_bart",\
     "ATT_DR_lasso_bart",\

    "ATT_outcome_ranf", "ATT_DR_ranf", "ATT_DR_adjust_ranf", "ATT_DR_weighting_ranf",\
     "ATT_DR_weighting_adjust_ranf", \

    "ATT_DR_ebal_ranf", "ATT_DR_cbps_ranf", "ATT_DR_opt_ranf", "ATT_DR_ebcw_ranf", \
    "ATT_DR_ps_ranf",\
    "ATT_DR_rf_ranf",  "ATT_DR_treebag_ranf", "ATT_DR_xgboost_ranf",\
     "ATT_DR_lasso_ranf"]

    for result in result_list:
        locals()[result] = np.array(locals()[result])
        locals()[result+"_bias"] = locals()[result].mean() - w
        locals()[result+"_sd"] = locals()[result].std()
        locals()[result+"_mae"] = (np.abs(locals()[result] -w)).mean()
        locals()[result+"_rmse"] = np.sqrt((np.square(locals()[result]-w)).mean())
        
        print(result+"_bias is", locals()[result+"_bias"])
        print(result+"_sd is", locals()[result+"_sd"])
        print(result+"_mae is", locals()[result+"_mae"])
        print(result+"_rmse is", locals()[result+"_rmse"])
        print("#################################################")
    

    ###### output the results

    # output the metrics.txt
    result_name = args.result_dir + "/class_"+ simu_class+"_d"+str(d)+"_n"+str(n)+\
        "_sc"+str(sc)+"_result_metrics.txt"
    with open(result_name, 'w') as metrics:
        for result in result_list:
            locals()[result] = np.array(locals()[result])
            locals()[result+"_bias"] = locals()[result].mean() - w
            locals()[result+"_sd"] = locals()[result].std()
            locals()[result+"_mae"] = (np.abs(locals()[result] -w)).mean()
            locals()[result+"_rmse"] = np.sqrt((np.square(locals()[result]-w)).mean())
            
            metrics.write(result+"_bias is %s \n" % locals()[result+"_bias"])
            metrics.write(result+"_sd is %s \n" % locals()[result+"_sd"])
            metrics.write(result+"_mae is %s \n" % locals()[result+"_mae"])
            metrics.write(result+"_rmse is %s \n" % locals()[result+"_rmse"])
            metrics.write("#################################################\n")


    # output the metrics.csv
    metrics_result = np.zeros((len(result_list), 4))
    result_name = args.result_dir + "/class_"+ simu_class+"_d"+str(d)+"_n"+str(n)+\
        "_sc"+str(sc)+"_result_metrics.csv"
    for i, result in enumerate(result_list):
        locals()[result] = np.array(locals()[result])
        locals()[result+"_bias"] = locals()[result].mean() - w
        locals()[result+"_sd"] = locals()[result].std()
        locals()[result+"_mae"] = (np.abs(locals()[result] -w)).mean()
        locals()[result+"_rmse"] = np.sqrt((np.square(locals()[result]-w)).mean())
        
        metrics_result[i] = np.array([locals()[result+"_bias"],\
            locals()[result+"_sd"], locals()[result+"_mae"], \
            locals()[result+"_rmse"]])
    
    metrics_out = pd.DataFrame(metrics_result, index = result_list, columns = \
        ["bias", "sd", "mae", "rmse"])
    metrics_out.to_csv(result_name, sep=',')


    # output the att.csv
    result_name = args.result_dir + "/class_"+ simu_class+"_d"+str(d)+"_n"+str(n)+\
        "_sc"+str(sc)+"_result_att.csv"

    att_out = pd.DataFrame({"ATT_base": ATT_base, "ATT_DLW": ATT_DLW, \
    "ATT_DLW_adjust": ATT_DLW_adjust, "ATT_weighting": ATT_weighting,\
    "ATT_weighting_adjust": ATT_weighting_adjust,\

    "ATT_ebal": ATT_ebal, \
    "ATT_cbps": ATT_cbps, "ATT_opt": ATT_opt, "ATT_ebcw": ATT_ebcw, \
    "ATT_ps": ATT_ps, "ATT_rf_weights": ATT_rf_weights,  \
    "ATT_treebag_weights":ATT_treebag_weights, "ATT_xgboost_weights":ATT_xgboost_weights,\
    "ATT_lasso_weights": ATT_lasso_weights, \

    "ATT_outcome_bart":ATT_outcome_bart, "ATT_DR_bart":ATT_DR_bart, \
    "ATT_DR_adjust_bart":ATT_DR_adjust_bart, "ATT_DR_weighting_bart":ATT_DR_weighting_bart,\
     "ATT_DR_weighting_adjust_bart": ATT_DR_weighting_adjust_bart, \

    "ATT_DR_ebal_bart": ATT_DR_ebal_bart, "ATT_DR_cbps_bart": ATT_DR_cbps_bart,\
     "ATT_DR_opt_bart": ATT_DR_opt_bart, "ATT_DR_ebcw_bart":ATT_DR_ebcw_bart, \
    "ATT_DR_ps_bart": ATT_DR_ps_bart,\

    "ATT_DR_rf_bart": ATT_DR_rf_bart,  \
     "ATT_DR_treebag_bart": ATT_DR_treebag_bart, "ATT_DR_xgboost_bart": ATT_DR_xgboost_bart,\
     "ATT_DR_lasso_bart":ATT_DR_lasso_bart,\

    "ATT_outcome_ranf":ATT_outcome_ranf, "ATT_DR_ranf": ATT_DR_ranf, "ATT_DR_adjust_ranf":ATT_DR_adjust_ranf,\
    "ATT_DR_weighting_ranf":ATT_DR_weighting_ranf, "ATT_DR_weighting_adjust_ranf":ATT_DR_weighting_adjust_ranf,\

    "ATT_DR_ebal_ranf": ATT_DR_ebal_ranf, "ATT_DR_cbps_ranf": ATT_DR_cbps_ranf,\
    "ATT_DR_opt_ranf": ATT_DR_opt_ranf, "ATT_DR_ebcw_ranf": ATT_DR_ebcw_ranf, \
    "ATT_DR_ps_ranf": ATT_DR_ps_ranf,\

    "ATT_DR_rf_ranf": ATT_DR_rf_ranf,  \
    "ATT_DR_treebag_ranf":ATT_DR_treebag_ranf, "ATT_DR_xgboost_ranf":ATT_DR_xgboost_ranf,\
    "ATT_DR_lasso_ranf": ATT_DR_lasso_ranf})
    
    att_out.to_csv(result_name,index=False,sep=',')
    
    end=time.time()
    time_use = (end-start)/3600
    print("Time_use is %.7f hours" % time_use)
