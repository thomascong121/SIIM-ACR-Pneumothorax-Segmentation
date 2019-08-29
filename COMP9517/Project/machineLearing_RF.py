import pandas as pd
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.color import rgb2gray

img = cv2.imread("/Users/congcong/Desktop/COMP9517/Project/COMP9517_Project/data-2/images/train-volume00.jpg")
print(img.shape)
plt.imshow("/Users/congcong/Desktop/COMP9517/Project/COMP9517_Project/data-2/images/train-volume00.jpg",'grey')


# #1.tiff, 2.0-1, 3.float32 4.tiffile

# img2 = cv2.imread("/Users/congcong/Desktop/COMP9517/Project/COMP9517_Project/data-2/output/0_predict.png")
# img2 = img2/255
# print(np.unique(img2))
# print(img2.shape)
