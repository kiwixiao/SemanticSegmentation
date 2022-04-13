#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 11:41:32 2021

@author: xiaz9n
"""
import os, glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
# %%
# define constants
# step 1  load and visualize the data
dataInputPath = './data/volumes'
imagePathInput = os.path.join(dataInputPath, 'img/')
maskPathInput = os.path.join(dataInputPath, 'mask/')

dataOutputPath = './data/slices/'
imageSliceOutput = os.path.join(dataOutputPath, 'img/')
maskSliceOutput = os.path.join(dataOutputPath, 'mask/')
if os.path.isdir(imageSliceOutput):
    shutil.rmtree(imageSliceOutput)
    print('previous version folder deleted')
    print('now recreate new empty image folder')
    os.mkdir(imageSliceOutput)
else:
    print('folder not exist, creating empty image folder')
    os.mkdir(imageSliceOutput)
if os.path.isdir(maskSliceOutput):
    shutil.rmtree(maskSliceOutput)
    print('previous version folder deleted')
    print('now recreate new empty mask folder')
    os.mkdir(maskSliceOutput)
else:
    print('folder not exist, creating empty mask folder')
    os.mkdir(maskSliceOutput)
    

# step 2 image normalization
HOUNSFIELD_MIN = 0
HOUNSFIELD_MAX = 2000
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN
# step 3 slicing and saving
SLICE_X = False
SLICE_Y = False
SLICE_Z = True

SLICE_DECIMATE_IDENTIFIER  = 3

''' load image and see max min image intensity range if the input is MR Images
'''
imgPath = os.path.join(imagePathInput, 'OSAMRI007_3301.nii')
img = nib.load(imgPath).get_fdata()
imgMin = np.min(img)
imgMax = np.max(img) 
imgShape = img.shape 
imgType = type(img)
# load image mask and see max min units
maskPath = os.path.join(maskPathInput, 'OSAMRI007_3301.nii.gz')
mask = nib.load(maskPath).get_fdata()
maskMin = np.min(mask)
maskMax = np.max(mask) 
maskShape = mask.shape 
maskType = type(mask)

'''show image slice
''' 
imgSlice = mask[:,:,59] # corol plane
plt.imshow(imgSlice, cmap='gray')
plt.show()

''' Normalize image
'''
def normalizeImageIntensityRange(img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE

nImg = normalizeImageIntensityRange(img)
normImgMin = np.min(nImg)
normImgMax = np.max(nImg)

'''Read image or mask volume
'''
def readImageVolume(imgPath, normalize=False):
    img = nib.load(imgPath).get_fdata()
    if normalize:
        return normalizeImageIntensityRange(img)
    else:
        return img
    
readImageVolume(imgPath, normalize=True)
readImageVolume(maskPath, normalize=False)

'''save volume slice to file
'''
def saveSlice(img, fname, path):
    img = np.uint8(img * 255)
    fout = os.path.join(path, f'{fname}.png')
    cv2.imwrite(fout, img)
    print(f'[+] Slice saved: {fout}', end='\r')
    
#saveSlice(nImg[:,:,60], 'test', imageSliceOutput)
#saveSlice(mask[:,:,60], 'test', maskSliceOutput)

''' Slice image in all directions and save
'''
def sliceAndSaveVolumeImage(vol, fname, path):
    (dimx, dimy, dimz) = vol.shape
    print(dimx, dimy, dimz)
    cnt = 0
    if SLICE_X:
        cnt += dimx
        print('Slicing X: ')
        for i in range(dimx):
            saveSlice(vol[i,:,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path)
            
    if SLICE_Y:
        cnt += dimy
        print('Slicing Y: ')
        for i in range(dimy):
            saveSlice(vol[:,i,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)
            
    if SLICE_Z:
        cnt += dimz
        print('Slicing Z: ')
        for i in range(dimz):
            saveSlice(vol[:,:,i], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path)
    return cnt

# Read and process image volumes
for index, filename in enumerate(sorted(glob.iglob(imagePathInput+'*.nii'))):
    fname = os.path.basename(filename)
    img = readImageVolume(filename, True)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, fname[0:9]+'_'+str(index), imageSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')

# Read and process image mask volumes
for index, filename in enumerate(sorted(glob.iglob(maskPathInput+'*.nii.gz'))):
    fname = os.path.basename(filename)
    img = readImageVolume(filename, False)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, fname[0:9]+'_'+str(index), maskSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')







