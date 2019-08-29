import cv2
import numpy as np
from matplotlib import pyplot as plt

def contrast_stretch(pixel,img_min,img_max,allow_max,allow_min):
	result = (pixel-img_min)*((allow_max-allow_min)/(img_max-img_min))+allow_min
	return int(result)


img = cv2.imread('SanFrancisco.jpg',0)
img_min = np.min(img)
img_max = np.max(img)

new_img = np.array([])
for row in img:
	new_row = np.array([])
	for col in row:
		contrast = contrast_stretch(col,img_min,img_max,255,0)
		new_row = np.append(new_row,contrast)
	if(new_img.size == 0):
		new_img = new_row.astype('uint8')
	else:
		new_img = np.vstack([new_img,new_row.astype('uint8')])


# print("max ",np.max(new_img))
# print("min ",np.min(new_img))

# print(type(new_img))
# print(type(img))

# print(new_img.dtype)
# print(img.dtype)

# print(new_img.shape)
# print(img.shape)

# print()
# print(new_img)
# print(img)

cv2.imshow('contrast image',new_img)
cv2.imshow('image',img)
cv2.waitKey(0)
img_list = [img,new_img]
title_list = ["Original image","Original image's histogram","Image after contrast_stretch","(Enhanced)Image's histogram"]
for i in range(2):
	plt.subplot(2,2,i+(i+1)),plt.imshow(img_list[i],'gray')
	plt.title(title_list[2*i])
	plt.subplot(2,2,i+(i+2)),plt.hist(img_list[i].ravel(),256,[0,256])
	plt.title(title_list[2*i+1])
plt.tight_layout()
plt.show()












