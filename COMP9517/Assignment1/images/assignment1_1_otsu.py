import cv2
import numpy as np
from matplotlib import pyplot as plt
def part1_otsu(img_path,median_blur=False,GaussianBlur=False):
	img = cv2.imread(img_path,0)
	if(median_blur):
		img= cv2.medianBlur(img,median_blur)
	if(GaussianBlur):
		img= cv2.GaussianBlur(img,(GaussianBlur[0],GaussianBlur[1]),GaussianBlur[2])


	ret1,th1=cv2.threshold(img,170,255,cv2.THRESH_BINARY)
	ret2,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#ret3,th3=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


	images=[img,th1,
	         img,th2]
	         #img,th3]
	titles =['original noisy image','global thresholding(v=127)',
	          'original noisy image',"otsu's thresholding"]
	          #'gaussian giltered image',"otus's thresholding"]
	for i in range(2):
	    plt.subplot(3,2,i*2+1),plt.imshow(images[i*2],'gray')
	    plt.title(titles[i*2]),plt.xticks([]),plt.yticks([])
	    plt.subplot(3,2,i*2+2),plt.imshow(images[i*2+1],'gray')
	    plt.title(titles[i*2+1]),plt.xticks([]),plt.yticks([])
	plt.subplot(3,2,5),plt.hist(img.ravel(),256)
	plt.show()