#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:08:08 2019

@author: malrawi
"""

import torchvision.transforms as transforms
from datasets import ImageDataset
import torch
from models import GeneratorResNet
from matplotlib import pyplot as plt

# A is the fashion image
# B is the pixel-level annotation

model_name = 'G_AB_60.pth' 
dataset_name = 'ClothCoParse'
experiment_name = '' # to be added 
path2model = './saved_models/ClothCoParse/'

print('model used', model_name) 

# loads a saved model
def get_GAN_AB_model(folder_model, model_name, device):          
    n_residual_blocks = 9 # this should be the same values used in training the G_AB model    
    G_AB = GeneratorResNet(input_shape=(3,0), num_residual_blocks = n_residual_blocks)        
    G_AB.load_state_dict(torch.load(folder_model + model_name,  map_location=device ),  )    
    
    if cuda: 
        G_AB = G_AB.to(device)
    return G_AB


# changed to display as color map and not image

def show_tensor(img, show_img=True):
    to_pil = transforms.ToPILImage() 
    img  = to_pil(img.squeeze()) # we can also use test_set[1121][0].numpy()    
    if show_img: 
        plt.imshow(img.convert('L'),  cmap= plt.cm.get_cmap("nipy_spectral"), vmin=0, vmax=255)
        # img.show()        
        # img.save('/home/malrawi/GAN_seg_img_414/'+'gg-col'+'.png') # can be used to save the image
    
    return img


transforms_used = transforms.Compose( [ transforms.ToTensor(), 
                  transforms.Normalize((0.5,0.5,0.5), (.5,.5,.5)) 
                 ] ) 

data_set = ImageDataset("../data/%s" % dataset_name, 
                            transforms_ = None, 
                            unaligned=False, 
                            mode='train' )

img_id=110 # getting some image, here index 100
PIL_A_img = data_set[img_id]['A']
PIL_B_img = data_set[img_id]['B']
real_A = transforms_used(PIL_A_img)  # tensor image

cuda = False # this will definetly work on the cpu if it is false
if cuda:
    cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

G_AB = get_GAN_AB_model(path2model, model_name,  device) # load the model
G_AB.eval()
# if not cuda:
#     G_AB=G_AB.cpu()

if cuda: real_A = real_A.to(device)
with torch.no_grad():
    B_output = G_AB(real_A.unsqueeze(0))

PIL_A_img.show() # show the original image            
plt.imshow(PIL_B_img.convert('L'),  cmap= plt.cm.get_cmap("nipy_spectral"), vmin=0, vmax=255) # show the pixel-level annotation
show_tensor(B_output) # show the segmented image we get from the model             

