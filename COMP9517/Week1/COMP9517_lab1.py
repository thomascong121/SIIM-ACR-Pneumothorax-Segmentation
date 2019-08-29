import numpy as np
import cv2
print(cv2.__version__)
img = cv2.imread("pict.jpg")#,1)
print(img.shape[:2])
#read an image
# img = cv2.imread('pict.jpg',1)
# #print image dim
# print(img.shape[:2])
# #display the image
# cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('image',img)
# #cv2.waitKey(0)
# #cv2.destroyAllWinows()

# #print a pixel value
# px = img[100,100]
# print(px)

# #get an ROI region'
# roi = img[100:120,100:120]

# #replace another region in the image with selcted ROI
# img[180:200,180:200] = roi

# cv2.imshow("image",img)
