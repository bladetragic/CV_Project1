import numpy as np 
import matplotlib as plt
from skimage import transform
import skimage.io as io
import skimage.color as color

#Convert RGB values to HSV values
def rgb_to_hsv(img_array):
    
    r,g,b = r/255.0, g/255.0, b/255.0 #Divide RGB values by 255 to normalize to values between 0 and 1
    # rgb_max = max(r,g,b)
    # rgb_min = min(r,g,b)
    # rgb_diff = rgb_max-rgb_min

    #Calculate V (value) for HSV color model
    v = np.max(img_array, axis = 2) 

    #Calculate S (saturation) for HSV color model; if V=0, then set S = 0
    if v == 0:
        s = 0
    else:
        c = v - np.min(img_array, axis=2)
        s = c/v

    if c == 0:
        h = 0
    elif v == r:
        pass





    
    # if rgb_diff == 0:
    #     h = 0
    # elif rgb_max == r:
    #     h = ( 60 * (0 + (g-b)/rgb_diff) % 360) #if max=R
    # elif rgb_max == g:
    #     h = ( 60 * (2 + (b-r)/(rgb_diff)) % 360) #if max=G
    # elif rgb_max == b:
    #     h = ( 60 * (4 + (r-g)/(rgb_diff)) % 360) #if max=B
    
    
 



#a = np.array([6,2,3,4,5,6])
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(a[1][1])