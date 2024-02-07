import numpy as np 
import matplotlib as plt
from skimage import transform
import skimage.io as io
import skimage.color as color

def parse_img_array(img):
    #print("Test\n")
    #print (img.shape)
    #print(img)
    print("\nTest\n")
    for arr in img:
        for pixels in arr:
            print(pixels)

            r,g,b = pixels[0], pixels[1], pixels[2]
            #print(r,g,b)

#Convert RGB values to HSV values
def rgb_to_hsv(img):

    #Parse Numpy array for r,g,b values
    for arr in img:
        for pixels in arr:
            #print(pixels)
            r,g,b = pixels[0]/255.0, pixels[1]/255.0, pixels[2]/255.0 #Store rgb channel values
            # pixels[0] = pixels[0]/255.0
            # pixels[1] = pixels[1]/255.0 
            # pixels[2] = pixels[2]/255.0
            # pixels[0] = r
            # pixels[1] = g 
            # pixels[2] = b
            # print(pixels[0], pixels[1], pixels[2])

            #Calculate V (value) for HSV color model
            #v = max(pixels[0], pixels[1], pixels[2])
            v = max(r,g,b)
            print("V: {}".format(v))

            #Calculate S (saturation) for HSV color model; if V=0, then set S = 0
            if v == 0:
                s = 0
            else:
                c = v - min(r,g,b)
                s = c/v
            print ("S: {}".format(s))

            #Calculate H (hue) for HSV color model
            if c == 0:
                h = 0
            elif v == r:
                h = (((g-b)/c) % 6)
            elif v == g:
                h = (((b-r)/c) + 2)
            elif v == b:
                h = (((r-g)/c) + 4)
            print ("H: {}".format(h))

            pixels[0], pixels[1], pixels[2] = h, s, v
    
    return(img)

    print(img)
    #r,g,b = r/255.0, g/255.0, b/255.0 #Divide RGB values by 255 to normalize to values between 0 and 1
    # rgb_max = max(r,g,b)
    # rgb_min = min(r,g,b)
    # rgb_diff = rgb_max-rgb_min

    #Calculate V (value) for HSV color model
    # v = np.max(img, axis = 2) 
    # #print("V:")
    # print ("V: {}".format(v))   

    #Calculate S (saturation) for HSV color model; if V=0, then set S = 0
    # if v == 0:
    #     s = 0
    # else:
    #     c = v - np.min(img, axis=2)
    #     s = c/v
    # print ("S: {}".format(s))
    # if c == 0:
    #     h = 0
    # elif v == r:
    #     pass
            #i = 0
            # for colors in pixels:
            #     print(pixels[i])
            #     i =  i + 1
        
        #print(arr)
        


#         0            1           2            3       
# 0 [ [[255, 0 , 0],[255, 0, 0],[255, 0 , 0],[255, 0 , 0]],
# 1   [[255, 0 , 0],[255, 0, 0],[255, 0 , 0],[255, 0 , 0]] ]        

# step-2 Define Image size - height, width and depth
# (rows, columns, dimensions) format
height, width, channel = 2, 3, 3

# step-3 Define Red,Green,Blue Color -for each- 0 to 255
red, green, blue = 255, 50, 50

# step-4 Generate RGB Numpy Array 
arr = np.full((height, width, channel), [red, green, blue], dtype=('uint8'))

# step-5 Show inline Image Plot
#plt.imshow(arr)

# step-6 Remove edge numbers from Image Plot
#plt.axis('off')

print(arr)

arr2 = rgb_to_hsv(arr)

print(arr2)




    
    # if rgb_diff == 0:
    #     h = 0
    # elif rgb_max == r:
    #     h = ( 60 * (0 + (g-b)/rgb_diff) % 360) #if max=R
    # elif rgb_max == g:
    #     h = ( 60 * (2 + (b-r)/(rgb_diff)) % 360) #if max=G
    # elif rgb_max == b:
    #     h = ( 60 * (4 + (r-g)/(rgb_diff)) % 360) #if max=B