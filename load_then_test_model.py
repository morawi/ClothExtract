#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:08:08 2019

@author: malrawi
"""

import torchvision.transforms as transforms
from datasets import ImageDataset
from torch.utils.data import DataLoader
import torch
from models import GeneratorResNet
from matplotlib import pyplot as plt

# from F1_loss import F1_loss_numpy, F1_loss_torch
# import numpy as np
# import time


# A is the fashion image
# B is the pixel-level annotation

path2model = './saved_models/ClothCoParse/'
model_name = 'G_AB_0.pth'
dataset_name = 'ClothCoParse' 
using_test_data = True
batch_test_size = 1 
channels = 3 
n_cpu = 0
input_shape = (channels, 0) # added 0 to resolve a problem

print('model used', model_name)





def get_GAN_AB_model(folder_model, model_name, cuda=False):  
        
    n_residual_blocks = 9 # this should be the same values used in training the G_AB model    
    G_AB = GeneratorResNet(input_shape, num_residual_blocks = n_residual_blocks)    
    if cuda: G_AB = G_AB.cuda()
    G_AB.load_state_dict(torch.load(folder_model + model_name ) )                 
    return G_AB
        
def show_tensor(img, show_img=True):
    to_pil = transforms.ToPILImage() 
    img1  = to_pil(img.cpu().squeeze()) # we can also use test_set[1121][0].numpy()    
    if show_img: 
        img1.show()        
#        img1.save('/home/malrawi/GAN_seg_img_414/'+'gg-col'+'.png')
    
    return img1


transforms_val = transforms.Compose( [ transforms.ToTensor(), 
                  transforms.Normalize((0.5,0.5,0.5), (.5,.5,.5)) 
                 ] ) 

data_set = ImageDataset("../data/%s" % dataset_name, 
                            transforms_ = None, 
                            unaligned=False, 
                            mode='train' )

img_id=110
PIL_A_img = data_set[img_id]['A']
PIL_B_img = data_set[img_id]['B']
real_A = transforms_val(PIL_A_img)  # tensor image
cuda = True if torch.cuda.is_available() else False
cuda = False
device = torch.device('cuda' if cuda else 'cpu')
G_AB = get_GAN_AB_model(path2model, model_name, cuda=cuda)
if not cuda:
    G_AB=G_AB.cpu()

if cuda: real_A = real_A.to(device)
with torch.no_grad():                                    
    B_gan = G_AB(real_A.unsqueeze(0))   

PIL_A_img.show()             
plt.imshow(PIL_B_img.convert('L'),  cmap= plt.cm.get_cmap("nipy_spectral"), vmin=0, vmax=255)
show_tensor(B_gan)              








''' Let's leave the below for now '''    

# val_dataloader = DataLoader(data_set,
#                             batch_size= batch_test_size, shuffle=False, 
#                             num_workers= n_cpu                            
#                             )

    
#test_GAN_AB_torch(path2model, model_name, val_dataloader, cuda)
# print('Using Torch based code to find F1')


# # this is used to measure the average time needed to translate an image
# x= [ i for i in range(len(f1))]
# plt.scatter(x, f1)
#            p_time= time.time()
#            for i in range(1000):
#                real_A = imgs_batch['A'].to(device) 
#                B_gan = G_AB(real_A) 
#                
#            print(time.time()-p_time)


'''
def test_GAN_AB_torch(folder_model, model_name, val_dataloader, use_cuda=False):      
    n_residual_blocks = 9 # this should be the same values used in training the G_AB model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G_AB = GeneratorResNet(num_residual_blocks=n_residual_blocks)
    cuda = True if torch.cuda.is_available() else False
    if cuda: G_AB = G_AB.cuda()
    G_AB.load_state_dict(torch.load(folder_model + model_name ) )
    F1 = []; R=[]; P=[]; print('\n -----------------')
        
    with torch.no_grad():
        for i, imgs_batch in enumerate(val_dataloader):               
            real_B = imgs_batch['B'].to(device)                                    
            B_gan = G_AB( imgs_batch['A'].to(device) )             

            # f1  = F1_loss_torch( B_gan, real_B, alpha = 10000, f1_inv=False ) # needed to threshold the GT as well                                                                
            # F1.append(f1.cpu().data.detach().numpy().tolist())
            # P.append(p)  # F1_loss_torch should be modified to give back p and rs, if needed
            # R.append(rs)
            
        # print(
        #         # 'R %.2f ' % (100*np.mean(R)), 
        #     #  'P %.2f ' % (100*np.mean(P)), 
        #       'F %.2f ' % (100*np.mean(F1))    
        #       )
            
    return real_B, B_gan # np.mean(F1) # , np.mean(P), np.mean(R)
'''
