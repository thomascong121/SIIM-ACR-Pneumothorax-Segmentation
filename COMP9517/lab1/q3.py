import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

img = cv2.imread('SanFrancisco.jpg',0)
sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)


# plt.imshow(sobelx,cmap = 'gray'),plt.title('x_direction')
# plt.savefig("x_direction.png",bbox_inches="tight")
# plt.imshow(sobely,cmap = 'gray'),plt.title('y_direction')
# plt.savefig("y_direction.png",bbox_inches="tight")


kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

dest_x = signal.convolve2d(img, kernel_y)
dest_x = np.absolute(dest_x)
print(dest_x)

# cv2.imshow('contrast image',dest_x)
# cv2.waitKey(0)
plt.imshow(dest_x,cmap = 'gray'),plt.title('x_direction')
plt.show()

#,cmap = 'gray'