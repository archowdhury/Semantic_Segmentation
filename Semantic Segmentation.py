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

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

#================================================================#
# DEFINE THE CONSTANTS
#================================================================#

IMG_WIDTH = 128
IMG_HEIGHT = 128
TRAIN_PATH = 'D:/Machine Learning/Computer Vision/U_NET/train/'
TEST_PATH = 'D:/Machine Learning/Computer Vision/U_NET/validation/'

len(next(os.walk(TRAIN_PATH))[1])

im0 = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))
im0.shape
im1 = cv2.imread(r'D:\Machine Learning\Computer Vision\U_NET\train\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e\masks\0fe691c27c3dcf767bc22539e10c840f894270c82fc60c8f0d81ee9e7c5b9509.png')
im2 = cv2.imread(r'D:\Machine Learning\Computer Vision\U_NET\train\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e\masks\1a88f51da11710786e4972b476f4b806e7feb984577610d4bad868e8fc26ad83.png')



cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)


result = (im1 + im2)
result = result.clip(0, 255).astype('uint8')

cv2.imshow('test', cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY))
cv2.waitKey(0)
cv2.destroyAllWindows()