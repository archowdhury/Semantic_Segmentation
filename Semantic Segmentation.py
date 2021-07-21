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
from tqdm import tqdm

warnings.filterwarnings('ignore')

#================================================================#
# DEFINE THE CONSTANTS
#================================================================#

IMG_WIDTH = 128
IMG_HEIGHT = 128
TRAIN_PATH = 'D:/Machine Learning/Computer Vision/U_NET/train/'
TEST_PATH = 'D:/Machine Learning/Computer Vision/U_NET/validation/'


#================================================================#
# READ IN THE IMAGES AND MASKS
#================================================================#

def load_images(folder, mode='test'):

    X, y = [], []

    for i in tqdm(os.listdir(folder)):
        # Read in the image file
        path, _, file = next(os.walk(folder + i + '/images'))
        image_file = path + '/' + file[0]
        img = cv2.imread(image_file)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        X.append(img)

        if mode == 'train':
            # Read in and concatenate the masks
            path, _, files = next(os.walk(folder + i + '/masks'))
            masks = [path + '/' + f for f in files]

            final_mask = np.zeros((IMG_WIDTH, IMG_HEIGHT, 1))
            for mask in masks:
                mask_img = cv2.imread(mask)
                mask_img = cv2.resize(mask_img, (IMG_WIDTH, IMG_HEIGHT))
                mask_img = np.expand_dims(cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY), axis=-1)
                final_mask += mask_img
            y.append(final_mask)

    return np.asarray(X), np.asarray(y)

X_train, y_train = load_images(TRAIN_PATH, mode='train')
X_test, _ = load_images(TEST_PATH, mode='test')


# Check if the images are loading properly
idx = 4
cv2.imshow('image', X_train[idx])
cv2.imshow('mask', y_train[idx])
cv2.waitKey(0)
cv2.destroyAllWindows()


