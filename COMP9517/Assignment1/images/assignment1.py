import cv2
import numpy as np
from matplotlib import pyplot as plt
def draw_plots(img,threshold_value,maxVal,adpt_mean=11,adpt_gaussian=11,C=2,median_blur=False,GaussianBlur=False):
	img = cv2.imread(img,0)
	if(median_blur):
		img= cv2.medianBlur(img,median_blur)
	elif(GaussianBlur):
		img= cv2.GaussianBlur(img,(GaussianBlur[0],GaussianBlur[1]),GaussianBlur[2])


	#global
	ret , thresh1 = cv2.threshold(img,threshold_value,maxVal,cv2.THRESH_BINARY)

	#otsu
	ret2,thresh4=cv2.threshold(img,0,maxVal,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	#adaptive
	thresh5 = cv2.adaptiveThreshold(img,maxVal,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY,adpt_mean,C)
	thresh6 = cv2.adaptiveThreshold(img,maxVal,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY,adpt_gaussian,C)
	titles = ['Binary','Otsu','Adaptive-mean','Adaptive-gaussian','histogram']
	images = [thresh1,thresh4,thresh5,thresh6]

	fig = plt.figure()
	counter = 1
	for i in range(4):
	    ax = plt.subplot(4,2 ,i+i+1)
	    plt.imshow(images[i],'gray')
	    plt.title(titles[i],fontsize=8)
	    plt.xticks([]),plt.yticks([])

	    plt.subplot(4,2 ,i+i+2)
	    plt.hist(images[i].ravel(),256,[0,256])
	    plt.title(titles[-1],fontsize=8)
	    plt.xticks([]),plt.yticks([])

	plt.show()
