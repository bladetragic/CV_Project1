# Brandon Bowles
# CSE 4310
# Assignment 1

import numpy as np 
import cv2
import color_space_test as cst
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.lib import stride_tricks

def random_crop(img, size):
    # Check if crop size is feasible
    if size > min(img.shape[0], img.shape[1]):
        raise ValueError("Crop size is larger than image dimensions")
    
    # Generate random center location
    center_x = np.random.randint(size//2, img.shape[1] - size//2)
    center_y = np.random.randint(size//2, img.shape[0] - size//2)
    
    # Calculate crop boundaries
    start_x = center_x - size//2
    end_x = start_x + size
    start_y = center_y - size//2
    end_y = start_y + size
    
    # Crop the image
    cropped_img = img[start_y:end_y, start_x:end_x]
    
    return cropped_img

def extract_patch(img, num_patches):

    # non-overlapping pathces
    patch_size = num_patches * num_patches

    height, width = img.shape[0], img.shape[1]

    shape = [height // patch_size, width // patch_size, 3] + [patch_size, patch_size]

    # (row, col, patch_row, patch_col)
    strides = [patch_size * s for s in img.strides[:2]] + list(img.strides)

    # extract patches
    patches = stride_tricks.as_strided(img, shape = shape, strides = strides)

    return patches

def resize_img(img, factor):
    # Check for valid factor value
    if factor <= 0 or not isinstance(factor, (float, int)):
        raise ValueError("ERROR: Scale factor must be a positive number")
    
    # Resize image using nearest neighbor interpolation
    img_resized = cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    
    return img_resized

def color_jitter(img, hue, saturation, value):
    
    # Convert RGB image to HSV
    hsv_img = cst.rgb_to_hsv_vectorized(img)

    # Define random perturbations for hue, saturation, and value
    delta_hue = np.random.uniform(-hue, hue)
    delta_saturation = np.random.uniform(-saturation, saturation)
    delta_value = np.random.uniform(-value, value)

    # Apply perturbations to HSV channels
    hsv_img[:, :, 0] = (hsv_img[:, :, 0] + delta_hue) % 180
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] + delta_saturation, 0, 1)
    hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] + delta_value, 0, 1)

    # Convert back to RGB
    jittered_img = cst.hsv_to_rgb_vectorized(hsv_img)

    return jittered_img