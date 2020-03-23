# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:13:40 2020

@author: malrawi
"""

import torchvision.transforms as transforms
from datasets import ImageDataset
import torch
from pix2pix_models import GeneratorUNet
from matplotlib import pyplot as plt
from PIL import Image

# A is the fashion image
# B is the pixel-level annotation

model_name = 'generator_200.pth' 
dataset_name = 'ClothCoParse'
experiment_name = '' # to be added 
path2model = 'C:/MyPrograms/p2p_model/'

print('model used', model_name) 

# loads a saved model
def get_GAN_AB_model(folder_model, model_name, device):          
    G_AB = GeneratorUNet()   
    G_AB.load_state_dict(torch.load(folder_model + model_name,  map_location=device ),  )    
    G_AB.eval()            
    return G_AB



transforms_used = transforms.Compose( 
    [ transforms.Resize((512, 512), Image.BICUBIC),
     transforms.ToTensor(), 
     transforms.Normalize((0.5,0.5,0.5), (.5,.5,.5)) 
                 ] ) 

data_set = ImageDataset("../data/%s" % dataset_name, 
                     transforms_=None, 
                     mode="train", 
                     unaligned=False, 
                     HPC_run=0, 
                     Convert_B2_mask = 0
                 )

cuda = False # this will definetly work on the cpu if it is false
if cuda:
    cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

G_AB = get_GAN_AB_model(path2model, model_name,  device) # load the model


i=0
while i<5:
    img_id=torch.randint( len(data_set), (1,)) # getting some image, here index 100
    PIL_A_img = data_set[img_id]['A']
    PIL_B_img = data_set[img_id]['B']
    real_A = transforms_used(PIL_A_img)  # tensor image
    
    
    if cuda: real_A = real_A.to(device)
    with torch.no_grad():
        B_output = G_AB(real_A.unsqueeze(0))
    
    PIL_A_img.show() # show the original image            
    plt.figure()
    plt.imshow(PIL_B_img.convert('L'),  cmap= plt.cm.get_cmap("nipy_spectral"), vmin=0, vmax=55) # show the pixel-level annotation
    plt.figure()
    plt.imshow(transforms.ToPILImage()(B_output.squeeze()).convert('L'),  cmap= plt.cm.get_cmap("nipy_spectral"), vmin=0, vmax=55)
    i+=1


# # changed to display as color map and not image

# def show_tensor(img, show_img=True):
#     to_pil = transforms.ToPILImage() 
#     img  = to_pil(img.squeeze()) # we can also use test_set[1121][0].numpy()    
#     if show_img: 
#         plt.imshow(img.convert('L'),  cmap= plt.cm.get_cmap("nipy_spectral"), vmin=0, vmax=55)
#         # img.show()        
#         # img.save('/home/malrawi/GAN_seg_img_414/'+'gg-col'+'.png') # can be used to save the image
    
#     return img


