import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
#1.slide: kernel size; threshold value 
#2.drop down:choose filtering;   choose question1/2;    choose image

def part2(img_path,threshold_value,maxVal,adpt_mean=25,adpt_gaussian=37,C=2,median_blur=False,GaussianBlur=False):
	img = cv2.imread(img_path,0)
	if(median_blur):
		img= cv2.medianBlur(img,median_blur)
		f,p,a,b,c = draw_plots(img,threshold_value,maxVal)
		
		#slide bar set up
		axcolor = 'lightgoldenrodyellow'
		Kernel_list = [0.2, 0.03, 0.5, 0.03]
		Thresh_list = [0.2, 0.0, 0.5, 0.03]
		axKernel = p.axes(Kernel_list, facecolor=axcolor)
		axThresh = p.axes(Thresh_list, facecolor=axcolor)
		sKernel = Slider(axKernel, 'Kernel',3, 17, valstep=2)
		sThresh = Slider(axThresh, 'threshold', 30, 200, valstep=10)
		#plt.show()

		def update(val):
			print("==========================")
			k = int(sKernel.val)
			t = int(sThresh.val)
			im = cv2.medianBlur(img,k)
			f1,p1,ax1,tlist,imlist = draw_plots(im,t,maxVal)
			p1.show()
			f.canvas.draw_idle()

		sKernel.on_changed(update)
		sThresh.on_changed(update)
		p.show()


	if(GaussianBlur):
		img= cv2.GaussianBlur(img,(GaussianBlur[0],GaussianBlur[1]),GaussianBlur[2])
		# plt.figure()
		# plt.imshow(img,'gray')
		# plt.show()
		print("=========== gaussian blur is   ",GaussianBlur)
		f,p,a,b,c = draw_plots(img,threshold_value,maxVal)
		
		#slide bar set up
		axcolor = 'lightgoldenrodyellow'
		Kernel_list = [0.2, 0.03, 0.5, 0.03]
		Thresh_list = [0.2, 0.0, 0.5, 0.03]
		axKernel = p.axes(Kernel_list, facecolor=axcolor)
		axThresh = p.axes(Thresh_list, facecolor=axcolor)
		sKernel = Slider(axKernel, 'Kernel',3, 17, valstep=2)
		sThresh = Slider(axThresh, 'threshold', 127, 200, valstep=10)
		plt.show()

		def update(val):
			print("==========================")
			k = int(sKernel.val)
			t = int(sThresh.val)

			im = cv2.GaussianBlur(img,k,GaussianBlur[1],GaussianBlur[2])
			f1,p1,ax1,tlist,imlist = draw_plots(im,t,maxVal)		
			p1.show()
			f.canvas.draw_idle()

		sKernel.on_changed(update)
		sThresh.on_changed(update)


def draw_plots(img,threshold_value,maxVal,adpt_mean=11,adpt_gaussian=11,C=2,median_blur=False,GaussianBlur=False):
	img = cv2.imread(img,0)
	if(median_blur):
		img= cv2.medianBlur(img,median_blur)
	elif(GaussianBlur):
		img= cv2.GaussianBlur(img,(GaussianBlur[0],GaussianBlur[1]),GaussianBlur[2])


	#global
	ret , thresh1 = cv2.threshold(img,threshold_value,maxVal,cv2.THRESH_BINARY)
	print("threshod set, ",threshold_value)
	#小于阈值的像素点灰度值不变，大于阈值的像素点置为该阈值
	# ret , thresh2 = cv2.threshold(img,threshold_value,maxVal,cv2.THRESH_TRUNC)
	# #小于阈值的像素点灰度值不变，大于阈值的像素点置为0,其中参数3任取
	# ret , thresh3 = cv2.threshold(img,threshold_value,maxVal,cv2.THRESH_TOZERO)(blurred)'original image',img,

	#otsu
	ret2,thresh4=cv2.threshold(img,0,maxVal,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	print('ret2 is ',ret2)

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
	#return fig,plt,ax,titles,images

def draw_plots_thresh(img,threshold_value,maxVal,adpt_mean=5,adpt_gaussian=5,C=2,median_blur=False,GaussianBlur=False):
	print('img',img)
	img = cv2.imread(img,0)
	if(median_blur):
		img1= cv2.medianBlur(img,median_blur)
	if(GaussianBlur):
		img2= cv2.GaussianBlur(img,(GaussianBlur[0],GaussianBlur[1]),GaussianBlur[2])
	plt.figure()
	plt.subplot(1,2,1)
	plt.imshow(img1,'gray')
	plt.title("mean")
	plt.subplot(1,2,2)
	plt.imshow(img2,'gray')
	plt.title("gaussian")
	plt.show()
	
def set_thre(img,median_blur=False,GaussianBlur=False):
	img = cv2.imread(img,0)
	if(median_blur):
		img= cv2.medianBlur(img,median_blur)
		print(1)
	elif(GaussianBlur):
		img= cv2.GaussianBlur(img,(GaussianBlur[0],GaussianBlur[1]),GaussianBlur[2])
	l = [40,50,60,70,80]
	for i in range(len(l)):
		ret , thresh1 = cv2.threshold(img,i,255,cv2.THRESH_BINARY)
		plt.subplot(1,5,i+1)
		plt.imshow(thresh1,'gray')
		plt.title("threhold = {0}".format(l[i]))
	plt.tight_layout()
	plt.show()







