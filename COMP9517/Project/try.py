import pandas as pd
import numpy as np 
import cv2,os
import matplotlib.pyplot as plt
import mahotas
from scipy import ndimage
from skimage.color import rgb2gray
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Features == X [30,n]
#feature extraction of images from labels? == Y[30,1]
#just label 0/1 for images from labels? threshold the label image, mark memebrane as 1, and rest as 0


#test-train split
img_dir = "/Users/congcong/Desktop/COMP9517/Project/data-2/images"
label_dir = "/Users/congcong/Desktop/COMP9517/Project/data-2/labels"
imageData = os.listdir(img_dir)
imageLabel = os.listdir(label_dir)
#ensure image and labels correspond
imageData.sort()
imageLabel.sort()
trainImage,validImage,trainLable,validLabel = train_test_split(imageData,imageLabel,test_size=0.2,shuffle=True,random_state=2)
def addPrefix(x):
    pre = x.split("-")[1][0:3]
    if(pre == "vol"):
        return img_dir + "/" + x
    elif(pre == "lab"):
        return label_dir + "/" + x
    else:
        return "Invalid input"
tmp = [trainImage,validImage,trainLable,validLabel]
trainImage = list(map(addPrefix,trainImage))
validImage = list(map(addPrefix,validImage))
trainLable = list(map(addPrefix,trainLable))
validLabel = list(map(addPrefix,validLabel))

def haralick_feature(image):
    #convert image into gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #compute the haralick textture feature vectror
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    #return the result
    return haralick


#step1: feature extraction

image_feature = []
label_feature = []
valid_feature = []
vlabel_feature = []

for i in trainImage:
    img = cv2.imread(i)
    h_feature = haralick_feature(img)
    image_feature.append(h_feature)
for i in trainLable:
    img = cv2.imread(i)
    h_feature = haralick_feature(img)
    label_feature.append(h_feature)
for i in validImage:
    img = cv2.imread(i)
    h_feature = haralick_feature(img)
    valid_feature.append(h_feature)
for i in validLabel:
    img = cv2.imread(i)
    h_feature = haralick_feature(img)
    vlabel_feature.append(h_feature)

image_feature = np.array(image_feature)
label_feature = np.array(label_feature)
valid_feature = np.array(valid_feature)
vlabel_feature = np.array(vlabel_feature)

print(image_feature.shape)
print(label_feature.shape)

model = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42)
model.fit(image_feature, label_feature.astype('int'))
result = model.predict(valid_feature)

print(result.shape)
print(vlabel_feature.shape)

# precision = metrics.precision_score(vlabel_feature.astype('int'), result.astype('int'), average='weighted')
# recall = metrics.recall_score(vlabel_feature.astype('int'), result.astype('int'), average='weighted')
# f1 = metrics.f1_score(vlabel_feature.astype('int'), result.astype('int'), average='weighted')
accuracy = metrics.accuracy_score(vlabel_feature, result)


#step2:fead feature and labels into ML algorithms
#step3:training
#step4:testing



# #1.tiff, 2.0-1, 3.float32 4.tiffile

# img2 = cv2.imread("/Users/congcong/Desktop/COMP9517/Project/COMP9517_Project/data-2/output/0_predict.png")
# img2 = img2/255
# print(np.unique(img2))
# print(img2.shape)


#adjusted_rand_score
#kappa statisic
#jaccard similarilty score