# Brandon Bowles
# CSE 4310
# Assignment 1

import os
import sys
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

# Read image from file
def read_image_from_file():
    if len(sys.argv) < 2:
        print("No filename provided")
        sys.exit(1)

    filename = sys.argv[1]
    
    return filename

# Save image
def save_file(file):

    filename = input('Please enter a file name for the modified file: ')
    
    file_location = input('Please enter the location where you would like to save the file: ')
    
    #Construct full file path
    full_file_path = os.path.join(file_location, filename)

    # Convert the numpy array to PIL Image
    img = Image.fromarray((file * 255).astype(np.uint8))
    img.save(full_file_path)

# Convert RGB values to HSV values
def rgb_to_hsv(img):

    #Normalize image array
    img_normalized = img/255.0

    #Create empty array
    arr2 = np.empty_like(img_normalized)

    #Parse Numpy array for r,g,b values
    for row in range(img.shape[0]): 
        for col in range(img.shape[1]):
            pixel = img_normalized[row, col]
            r,g,b = pixel[0], pixel[1], pixel[2] #Store rgb channel values
            print(f"R, G, B Values: {r} - {g} - {b}")

            #Calculate V (value) for HSV color model
            v = max(r,g,b)
            print(f"V: {v}")

            #Calculate C (chroma) value
            c = v - min(r,g,b)

            #Calculate S (saturation) for HSV color model; if V=0, then set S = 0
            if v == 0:
                s = 0
            else:
                s = c/v
            print (f"S: {s}")

            #Calculate H (hue) for HSV color model
            if c == 0:
                h = 0
            elif v == r:
                h = (((g-b)/c) % 6)
            elif v == g:
                h = (((b-r)/c) + 2)
            elif v == b:
                h = (((r-g)/c) + 4)

            h = h*60

            print (f"H: {h}")


            #Input HSV values into empty array
            arr2[row, col] = [h, s, v]

    return arr2   

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

# Modify hsv image
def modify_hsv_image(hsv_img, hue_mod_value, sat_mod_value, val_mod_value):
    
    # Clamp hue modification value to [0, 360]
    #hue_mod_value = max(0, min(hue_mod_value, 360))

    # Check if saturation and value modifications are within range
    # if not (0 <= sat_mod_value <= 1) or not (0 <= val_mod_value <= 1):
    #     print("Saturation and value modification values should be in the range of [0, 1].")
    #     sys.exit(1)

    # Modify HSV values
    hsv_img[:, :, 0] = np.clip(hsv_img[:, :, 0] + hue_mod_value, 0, 360) 
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] + sat_mod_value, 0, 1)
    hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] + val_mod_value, 0, 1)   

    #Convert image back to RGB
    modified_img = hsv_to_rgb_vectorized(hsv_img)
    
    return modified_img

#Read in image
filename = read_image_from_file() 
print(f'File name: {filename}')
img = plt.imread(filename)

#Store hue, saturation, and value modification values from command line
hue_mod_value = float(sys.argv[2])
sat_mod_value = float(sys.argv[3])
val_mod_value = float(sys.argv[4])

#Display RGB image
plt.imshow(img)
plt.axis('off')
plt.title('Tree')
plt.show()

#Convert RGB image to HSV image
img2 = rgb_to_hsv_vectorized(img)

#img3 = hsv_to_rgb_vectorized(img2)

#Display HSV image
plt.imshow(img2)
plt.axis('off')
plt.title('Tree')
plt.show()

#Modify HSV values for HSV image
img3 = modify_hsv_image(img2, hue_mod_value, sat_mod_value, val_mod_value)

#Display RGB image
print(img3.shape)
plt.imshow(img3)
plt.axis('off')
plt.title('Tree')
plt.show()

save_file(img3)



