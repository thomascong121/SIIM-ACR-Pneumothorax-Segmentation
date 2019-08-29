import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
#do convolution by ourself
img = cv2.imread('SanFrancisco.jpg',0)
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])

result = signal.convolve2d(img,kernel)
result = np.absolute(result)
#check with package 
laplacian = cv2.Laplacian(img,cv2.CV_8U) 

#canny
canny1 = cv2.Canny(img,20,100)
canny2 = cv2.Canny(img,50,150)

plt.subplot(2,2,1),plt.imshow(result,cmap = 'gray')
plt.title('laplacian_convol')
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('laplacian_package')
plt.subplot(2,2,3),plt.imshow(canny1,cmap = 'gray')
plt.title('Canny1')
plt.subplot(2,2,4),plt.imshow(canny2,cmap = 'gray')
plt.title('Canny2')
plt.tight_layout()
plt.show()
