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

            h = h*60

            print (f"H: {h}")


            #Input HSV values into empty array
            arr2[row, col] = [h, s, v]

    return(arr2)   

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
    #print (hsv_img)
    return(hsv_img)   


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
    #print(x.shape)

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
    
    # print(rprime)
    # print(gprime)
    # print(bprime)

    #Calculate m values
    m = v - c

    #Calculate final R, G, B values
    r, g, b = (rprime + m, gprime + m, bprime + m)

    # print(r)
    # print(g)
    # print(b)

    rgb_img = np.stack((r, g, b), axis=2)

    return(rgb_img)

#Read in image
img = plt.imread(r"C:\Users\blade\Pictures\Image_Datasets\misc\4.1.06.tiff")

print(img.shape)

#Display RGB image
plt.imshow(img)
plt.axis('off')
plt.title('Tree')
plt.show()

#img2 = rgb_to_hsv(img)
img2 = rgb_to_hsv_vectorized(img)

img3 = hsv_to_rgb_vectorized(img2)

#Display HSV image
plt.imshow(img2)
plt.axis('off')
plt.title('Tree')
plt.show()

#Display RGB image
# print(img3)
plt.imshow(img3)
plt.axis('off')
plt.title('Tree')
plt.show()



