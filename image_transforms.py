import sys
import numpy as np 
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

#Read in image
img = plt.imread(r"C:\Users\blade\Pictures\Image_Datasets\misc\4.1.06.tiff")

img2 = random_crop(img, 80)

img3 = extract_patch(img, 4)

print(img3.shape)

plt.imshow(img3[:,:,:,0,0])
plt.axis('off')
plt.title('Tree')
plt.show()