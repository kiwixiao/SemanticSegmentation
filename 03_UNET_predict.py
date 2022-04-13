#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 19:28:58 2022

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
from tensorflow import keras
#%%
data_dir = 'data/slices/'
data_dir_test = os.path.join(data_dir, 'test')
# The images should be stored under: "data/slices/test/img/img"
data_dir_test_image = os.path.join(data_dir_test, 'img')
# The images should be stored under: "data/slices/test/mask/img"
data_dir_test_mask = os.path.join(data_dir_test, 'mask')

test_images = []
for directory_path in glob.glob(data_dir_test_image+'/img'):
    for img_path in glob.glob(os.path.join(directory_path,'*.png')):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        #train_labels.append(label)
#Convert list to array for machine learning processing
test_images = np.array(test_images)
test_masks = []
for directory_path in glob.glob(data_dir_test_mask+'/img'):
    for mask_path in glob.glob(os.path.join(directory_path,'*.png')):
        print(mask_path)
        mask = cv2.imread(mask_path, 0)
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_masks.append(mask)
        #train_labels.append(label)
#Convert list to array for machine learning processing
test_masks = np.array(test_masks)
truth = np.expand_dims(test_masks, axis=-1)



model = keras.models.load_model('Unet_vgg16_batch_size_32_epochs_500.h5', compile=False)
model.summary()
prediction = model.predict(test_images)

#%%
# view and save segmented image
index = np.random.choice(len(test_images))
prediction_image = prediction[index].reshape(mask.shape)
plt.figure()
plt.subplot(131)
plt.imshow(test_images[index], cmap='gray')
plt.subplot(132)
plt.imshow(prediction_image, cmap='gray')
plt.subplot(133)
plt.imshow(test_masks[index], cmap='gray')
plt.imsave('predict_mask_demo.png', prediction_image, cmap='gray')





