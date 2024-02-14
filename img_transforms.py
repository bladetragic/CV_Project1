import sys
import numpy as np 
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import transform
import skimage.io as io
import skimage.color as color
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

    # # Calculate the new dimensions
    # new_width = int(img.shape[1] * factor)
    # new_height = int(img.shape[0] * factor)

    # # Resize the image using nearest neighbor interpolation
    # img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Resize image using nearest neighbor interpolation
    img_resized = cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    
    return img_resized

def color_jitter(img, hue, saturation, value):
    
    # Convert RGB image to HSV
    hsv_img = rgb_to_hsv_vectorized(img)

    # Define random perturbations for hue, saturation, and value
    delta_hue = np.random.uniform(-hue, hue)
    delta_saturation = np.random.uniform(-saturation, saturation)
    delta_value = np.random.uniform(-value, value)

    # Apply perturbations to HSV channels
    hsv_img[:, :, 0] = (hsv_img[:, :, 0] + delta_hue) % 180
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] + delta_saturation, 0, 1)
    hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] + delta_value, 0, 1)

    # Convert back to RGB
    jittered_img = hsv_to_rgb_vectorized(hsv_img)

    return jittered_img

# Convert RGB values to HSV values - vectorized operations
def rgb_to_hsv_vectorized(img):
   
    #Normalize image array
    img_normalized = img/255.0

    #Split R, G, B channels into separate ndarrays
    r, g, b = img_normalized[..., 0], img_normalized[..., 1], img_normalized[..., 2]
   
    #Calculate V (value)
    v = np.max(img_normalized, axis=2)

    #Calculate C (chroma)
    c = v - np.min(img_normalized, axis=2)


    #Calculate S (saturation)
    s = np.where(v == 0, 0.0, c/v)

    #Create zero array for hprime values 
    hprime = np.zeros_like(v)

    #Calculate H values based on piecewise function
    hprime = np.where(v == r, ((g-b)/c) % 6, hprime)
    hprime = np.where(v == g, ((b-r)/c) + 2, hprime)
    hprime = np.where(v == b, ((r-g)/c) + 4, hprime)
    hprime = np.nan_to_num(hprime)

    #Scale H value to [0, 360]
    hprime = hprime * 60
    
    #Combine H,S,V values into one ndarray
    hsv_img = np.stack((hprime, s, v), axis=2)
    
    return hsv_img   

# Convert HSV values to RGB values - vectorized operations
def hsv_to_rgb_vectorized(img):

    #Split up H, S, V channels into separate ndarrays
    h, s, v = img[..., 0], img[..., 1], img[..., 2]

    #Calculate C (Chroma) value
    c = v * s

    #Calculate H' values
    hprime = h/60

    #Create array of ones to perform x value calculation
    ones = np.ones_like(hprime)
    x = c * (ones - abs((hprime % 2) - ones))

    #Calculate R', G', B' values
    # R', G', B' if 0 <= H' < 1
    rprime = np.where((hprime >= 0) & (hprime < 1), c, 0)
    gprime = np.where((hprime >= 0) & (hprime < 1), x, 0)
    bprime = np.where((hprime >= 0) & (hprime < 1), 0, 0)

    # R', G', B' if 1 <= H' < 2
    rprime = np.where((hprime >= 1) & (hprime < 2), x, rprime)
    gprime = np.where((hprime >= 1) & (hprime < 2), c, gprime)
    bprime = np.where((hprime >= 1) & (hprime < 2), 0, bprime)

    # R', G', B' if 2 <= H' < 3
    rprime = np.where((hprime >= 2) & (hprime < 3), 0, rprime)
    gprime = np.where((hprime >= 2) & (hprime < 3), c, gprime)
    bprime = np.where((hprime >= 2) & (hprime < 3), x, bprime)

    # R', G', B' if 3 <= H' < 4
    rprime = np.where((hprime >= 3) & (hprime < 4), 0, rprime)
    gprime = np.where((hprime >= 3) & (hprime < 4), x, gprime)
    bprime = np.where((hprime >= 3) & (hprime < 4), c, bprime)

    # R', G', B' if 4 <= H' < 5
    rprime = np.where((hprime >= 4) & (hprime < 5), x, rprime)
    gprime = np.where((hprime >= 4) & (hprime < 5), 0, gprime)
    bprime = np.where((hprime >= 4) & (hprime < 5), c, bprime)

    # R', G', B' if 5 <= H' < 6
    rprime = np.where((hprime >= 5) & (hprime < 6), c, rprime)
    gprime = np.where((hprime >= 5) & (hprime < 6), 0, gprime)
    bprime = np.where((hprime >= 5) & (hprime < 6), x, bprime)
    
    #Calculate m values
    m = v - c

    #Calculate final R, G, B values
    r, g, b = (rprime + m, gprime + m, bprime + m)

    #Combine values to create final RGB image array
    rgb_img = np.stack((r, g, b), axis=2)

    return rgb_img

#Read in image
img = plt.imread(r"C:\Users\blade\Pictures\Image_Datasets\misc\4.1.06.tiff")

#Display original image
plt.imshow(img)
plt.axis('off')
plt.title('Tree')
plt.show()

# Crop image
#cropped_img = random_crop(img, 80)

# Extract patches from image
#img_patch = extract_patch(img, 4)

#Display extracted patch
# plt.imshow(img_patch[:,:,:,0,0])
# plt.axis('off')
# plt.title('Tree')
# plt.show()

# # Resize image
# resized_img = resize_img(img, 0.2)

# # Display resized image
# plt.imshow(resized_img)
# plt.axis('off')
# plt.title('Tree')
# plt.show()

jitter_img = color_jitter(img, 50, 0.5,0.5)

# Display jitter image
plt.imshow(jitter_img)
plt.axis('off')
plt.title('Tree')
plt.show()