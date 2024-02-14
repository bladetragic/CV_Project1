# Brandon Bowles
# CSE 4310
# Assignment 1

import os
import numpy as np 
import img_transforms as it
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.lib import stride_tricks
from PIL import Image

def create_image_pyramid(img, pyramid_height, filename):
    
    # Create directory to save pyramid images
    directory = filename.split('.')[0] + "_pyramid"
    os.makedirs(directory, exist_ok=True)

    # Save the original image
    original_file_path = os.path.join(directory, filename)
    original_img = Image.fromarray(img)
    original_img.save(original_file_path)

    # Iterate over every level of pyramid and save resized image files
    for level in range(1, pyramid_height + 1):
        scale_factor = 2 ** level
        img_resized = it.resize_img(img, 1 / scale_factor)
        new_filename = f"{filename.split('.')[0]}_{scale_factor}x.png"
        full_file_path = os.path.join(directory, new_filename)
        resized_img_pil = Image.fromarray(img_resized)
        resized_img_pil.save(full_file_path)

#img1 = plt.imread(r"C:\Users\blade\Pictures\Image_Datasets\misc\4.1.06.tiff")

#create_image_pyramid(img1, 4, 'tree.png')