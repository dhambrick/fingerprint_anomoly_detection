import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt 
import glob 
from PIL import Image
import torch
import torchvision
from tqdm import tqdm
import pickle 
#Get list of filenames
# Specify data root path
data_root = '.\socofing\SOCOFing\Real'
#Collect list of image files
data_filenames = glob.glob(data_root+'\*.bmp')
metadata =  [x  .rsplit('\\',1)[1]
                .split('.')[0]
                .replace('__','_')
                .split('_')[0:4] for x in data_filenames]
metadata_labels = ['sub_idx','gender','hand','finger']
meta_df = pd.DataFrame(metadata,columns=metadata_labels)
data = [] 
pt_centercrop_transform_rectangle = torchvision.transforms.CenterCrop((64,64))
new_size = (64,64)
print('Parsing and Cropping input images')
for fname in tqdm(data_filenames):
    im = Image.open(fname).convert(mode='1')
    img_resize = im.resize(new_size)
    img_resize = np.array(img_resize)
    data.append(img_resize.flatten())
print(im.size)
print(len(data))
#print("Serializing Transformed Images")
#pickled_data = open('cropped_imgs','ab')
#pickle.dump(data,pickled_data)
#pickled_data.close()
#print("Pickling Complete")
#im.show()
#img_resize.show()
