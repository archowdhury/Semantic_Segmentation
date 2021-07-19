#================================================================#
# IMPORT THE PACKAGES
#================================================================#
import os
import random
import warnings
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from functools import reduce

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

#================================================================#
# DEFINE THE CONSTANTS
#================================================================#

IMG_WIDTH = 128
IMG_HEIGHT = 128
TRAIN_PATH = 'D:/Machine Learning/Computer Vision/U_NET/train/'
TEST_PATH = 'D:/Machine Learning/Computer Vision/U_NET/validation/'


curdir = os.path.join(TRAIN_PATH, os.listdir(TRAIN_PATH)[6])

for (dirpath, dirname, filenames) in os.walk(curdir):
    if 'images' in dirpath:
       image = os.path.join(dirpath, filenames[0])
    if 'masks' in dirpath:
        masks = [os.path.join(dirpath, file) for file in filenames]

image = read_image(image)

final_mask = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))
for i in masks:
    final_mask += read_image(i)

cv2.imshow('Image', image)
cv2.imshow('Mask', final_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()



