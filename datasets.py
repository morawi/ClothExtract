import glob
import random
import os
import scipy.io as sio

from torch.utils.data import Dataset # Dataset class from PyTorch
from PIL import Image # PIL is a nice Python Image Library that we can use to handle images
import torchvision.transforms as transforms # torch transform used for computer vision applications

import matplotlib.pyplot as plt





# convert the image to RGB in case it has only one channel
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image




class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        # mode "train" during learning / training and "test" during testing
        # we will have two folders; one called train and the other test; hence, we load the images according to whay we are doing (train or test?)
        # root is the folder that contains the data
        # transform_ is an actual parameter that contains some transform that we can apply on each image, for example, rotation, translation, scaling, etc
        # if the source and target are aligned, this is supervised learning, otherwise it is unsupervised learning
        # Yes, amazingly, CycleGan can learn well even if the source and target images are unaligned (ie umpaired)
        self.transform = transforms.Compose(transforms_) # image transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*")) # get the source image file-names
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*")) # get the target image file-names

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]) # read the image, according to the file name, index select which image to read; index=1 means get the first image in the list self.files_A

        if self.unaligned:
            annot = sio.loadmat(self.files_B[random.randint(0, len(self.files_B) - 1)])
            image_B = Image.fromarray(annot["groundtruth"])
        else:
            annot = sio.loadmat(self.files_B[index % len(self.files_B)])
            image_B = Image.fromarray(annot["groundtruth"])

#        Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A) # here we apply the transform on the source
        item_B = self.transform(image_B) # apply the transform on the target (in our case, the target is the pixel-wise annotation that marks the garments)
        return {"A": item_A, "B": item_B} # we are returning both the source and the target

    def __len__(self): # this function returns the length of the dataset, the source might not equal the target if the data is unaligned
        return max(len(self.files_A), len(self.files_B))
# NB. Done on the fly, have not therefore checked it for spelling mistakes

''' here data folder is one level behind the code folder, as we want to separate the code from data
inside data folder there is train_folder 
should have two sub-folders, A (contains the images) and B (containes the annotations) '''

x_data = ImageDataset("../data/%s" % "ClothCoParse",  
                           transforms_='',                            
                           unaligned=False, 
                           mode = "train",                           
                           )

x_data[0]  #accessing the first element in the data, should have the first image and its corresponding pixel-levele annotation
img = x_data[0]['A']  # getting the image
anno = x_data[0]['B']  # getting the annotation


plt.imshow(anno.convert('L'),  cmap= plt.cm.get_cmap("gist_stern"), vmin=0, vmax=255)


