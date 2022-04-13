#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:30:19 2021

@author: xiaz9n
"""

import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import segmentation_models as sm
sm.set_framework('tf.keras')
import glob
import cv2
import seaborn as sns
import numpy as np
# %%Define constants
SEED = 909
batch_size = 2
epochs = 10
# resize image size, but fully conv model is fine without resize
SIZE_X = 224
SIZE_Y = 224
n_classes = 1
# EPOCH_STEP_TRAIN = NUM_TRAIN // BATCH_SIZE_TRAIN
# EPOCH_STEP_TEST = NUM_TEST // BATCH_SIZE_TEST

BACKBONE='vgg19'
preprocess_input = sm.get_preprocessing(BACKBONE)

#%%
data_dir = 'data/slices/'
data_dir_train = os.path.join(data_dir, 'training')
# The images should be stored under: "data/slices/training/img/img"
data_dir_train_image = os.path.join(data_dir_train, 'img')
# The images should be stored under: "data/slices/training/mask/img"
data_dir_train_mask = os.path.join(data_dir_train, 'mask')

data_dir_test = os.path.join(data_dir, 'test')
# The images should be stored under: "data/slices/test/img/img"
data_dir_test_image = os.path.join(data_dir_test, 'img')
# The images should be stored under: "data/slices/test/mask/img"
data_dir_test_mask = os.path.join(data_dir_test, 'mask')

#%%

def display(display_list):
    plt.figure(figsize=(15,15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
    plt.show()

def show_dataset(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        display([image[10], mask[10]])

def show_prediction(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        pred_mask = model.predict(image)[0] > 0.5
        display([image[0], mask[0], pred_mask])

#%%
# initialize lsit for train data
train_images = []
for directory_path in glob.glob(data_dir_train_image+'/img'):
    for img_path in glob.glob(os.path.join(directory_path,'*.png')):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        #train_labels.append(label)
#Convert list to array for machine learning processing
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = []
for directory_path in glob.glob(data_dir_train_mask+'/img'):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)
        #mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        train_masks.append(mask)
        #train_labels.append(label)
#Convert list to array for machine learning processing
train_masks = np.array(train_masks)

#Use customary x_train and y_train variables
X = train_images
Y = train_masks
# Y = np.expand_dims(Y, axis=3) #May not be necessary.. leftover from previous code
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

y_train = tf.cast(y_train, tf.float32)
y_val = tf.cast(y_val, tf.float32)


# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze=True)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

model.compile(optimizer='adam', loss=sm.losses.binary_focal_dice_loss, metrics=sm.metrics.IOUScore(threshold=0.5))

print(model.summary())


history=model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))
model.save('Unet_'+BACKBONE+'_batch_size_{}_epochs_{}.h5'.format(batch_size, epochs))


#accuracy = model.evaluate(x_val, y_val)
#plot the training and validation accuracy and loss at each epoch
# Plot training & validation iou_score values
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')

# Plot training & validation loss values
plt.subplot(212)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')
plt.savefig('Unet_'+BACKBONE+'_performance_curve_batchsize_{}_epochs_{}.png'.format(batch_size, epochs))
plt.show()
plt.close()









