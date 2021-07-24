# ================================================================#
# IMPORT THE PACKAGES
# ================================================================#
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.morphology import label
from tqdm import tqdm
from PIL import Image

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, MaxPooling2D, Dropout, Lambda, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

warnings.filterwarnings('ignore')

# ================================================================#
# DEFINE THE CONSTANTS
# ================================================================#

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = 'D:/Machine Learning/Computer Vision/U_NET/train/'
TEST_PATH = 'D:/Machine Learning/Computer Vision/U_NET/validation/'


# ================================================================#
# READ IN THE IMAGES AND MASKS
# ================================================================#

def load_images(folder, mode='test'):
    num_items = len(os.listdir(folder))

    X = []
    y = []

    for idx, i in tqdm(enumerate(os.listdir(folder)), total=num_items):

        # Read in the image file
        path, _, file = next(os.walk(folder + i + '/images'))
        image_file = path + '/' + file[0]
        img = Image.open(image_file)
        img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.NEAREST)
        numpydata = np.asarray(img)[:, :, :IMG_CHANNELS]
        X.append(numpydata)

        if mode == 'train':
            path, _, files = next(os.walk(folder + i + '/masks'))
            mask_files = [path + '/' + f for f in files]

            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH))
            for f in mask_files:
                submask = Image.open(f)
                submask = submask.resize((IMG_HEIGHT, IMG_WIDTH))
                mask = np.maximum(mask, np.array(submask))
            y.append(mask)

    # Convert X to array
    X = np.asarray(X)

    # Convert y to a boolean array
    y = np.asarray(y)
    y = y > 0

    return X, y

# Load the train and test images
X_train, y_train = load_images(TRAIN_PATH, mode='train')
X_test, _ = load_images(TEST_PATH, mode='test')


# ================================================================#
# DEFINE THE CUSTOM IOU LOSS METRIC
# ================================================================#

def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)


def my_iou_metric(label, pred):
    metric_value = tf.py_function(iou_metric_batch, [label, pred], tf.float32)
    return metric_value


# ================================================================#
# DEFINE THE U-NET MODEL
# ================================================================#

filter_size = (3, 3)
pooling_size = (2, 2)
strides = (2, 2)

# input
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255)(inputs)

# contraction path layers
c1 = Conv2D(16, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D(pooling_size)(c1)

c2 = Conv2D(32, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D(pooling_size)(c2)

c3 = Conv2D(64, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D(pooling_size)(c3)

c4 = Conv2D(128, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPooling2D(pooling_size)(c4)

c5 = Conv2D(256, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(c5)

# expansion path layers
u6 = Conv2DTranspose(128, (2, 2), strides=strides, padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=strides, padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=strides, padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=strides, padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, filter_size, activation='elu', kernel_initializer='he_normal', padding='same')(c9)

# output
outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

# create the final model
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

model.summary()

# ================================================================#
# TRAIN THE MODEL
# ================================================================#

earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

checkpoint = ModelCheckpoint("nuclei_semantic_segmentation_model.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

results = model.fit(X_train, y_train,
                    validation_split=0.1,
                    batch_size=16,
                    epochs=10,
                    callbacks=[earlystop, checkpoint])

# ================================================================#
# PREDICT THE MASKS FOR THE TRAINING SET
# ================================================================#

y_train_pred = model.predict(X_train, verbose=1)
y_train_pred = (y_train_pred > 0.5).astype(np.uint8)

idx = 12
plt.imshow(X_train[idx])
plt.imshow(y_train[idx])
plt.imshow(y_train_pred[idx])

