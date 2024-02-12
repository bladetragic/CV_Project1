import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import transform
import skimage.io as io
import skimage.color as color

def parse_img_array(img):
    for arr in img:
        for pixels in arr:
            print(pixels)

            r,g,b = pixels[0], pixels[1], pixels[2]
            #print(r,g,b)

#Convert RGB values to HSV values
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
            print (f"H: {h}")

            #Input HSV values into empty array
            arr2[row, col] = [h, s, v]

    # Scale H, S, V values to the range [0, 255]
    arr2[:, :, 0] *= 360  # Scale H to [0, 360]
    arr2[:, :, 1:] *= 255  # Scale S and V to [0, 255]

    # Round and convert to uint8
    arr2 = np.round(arr2).astype(np.uint8)

    return(arr2)   

def hsv_to_rgb(img):
    pass

#         0            1           2            3       
# 0 [ [[255, 0 , 0],[255, 0, 0],[255, 0 , 0],[255, 0 , 0]],
# 1   [[255, 0 , 0],[255, 0, 0],[255, 0 , 0],[255, 0 , 0]] ]        

img = plt.imread(r"C:\Users\blade\Pictures\Image_Datasets\misc\4.1.06.tiff")

print(img.shape)

plt.imshow(img)
plt.axis('off')
plt.title('Tree')
plt.show()

arr2 = rgb_to_hsv(img)

#print("ARRAY 1")
#print(img)
print(arr2)
plt.imshow(arr2)
plt.axis('off')
plt.title('Tree')
plt.show()


