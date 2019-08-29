"""
COMP9517 Lab 04, Week 6
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

from scipy import ndimage as ndi
from skimage.morphology import watershed,erosion,dilation
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift
from itertools import cycle
from PIL import Image

size = 100, 100

img_dir = "data"
ext_dir = "ext_data"

img_names = ["orange_half.png", "two_halves_binary.png"]
ext_names = ["coins.png", "two_halves.png"]

images = [img_dir + "/" + i for i in img_names]
ext_images = [ext_dir + "/" + i for i in ext_names]


def plot_three_images(figure_title, image1, label1,
                      image2, label2, image3, label3):
    fig = plt.figure()
    fig.suptitle(figure_title)

    # Display the first image
    fig.add_subplot(1, 3, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)

    # Display the second image
    fig.add_subplot(1, 3, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)

    # Display the third image
    fig.add_subplot(1, 3, 3)
    plt.imshow(image3)
    plt.axis('off')
    plt.title(label3)

    plt.show()


# for img_path in images:
#     img = Image.open(img_path)
#     #缩略图
#     img.thumbnail(size)  # Convert the image to 100 x 100
#     # Convert the image to a numpy matrix
#     img_mat = np.array(img)[:, :, :3]
#     original_shape = img_mat.shape
#     #
#     # +--------------------+
#     # |     Question 1     |
#     # +--------------------+
#     #
#     # TODO: perform MeanShift on image
#     # Follow the hints in the lab spec.

#     # Step 1 - Extract the three RGB colour channels
#     # Hint: It will be useful to store the shape of one of the colour
#     # channels so we can reshape the flattened matrix back to this shape.
#     #print("step 1: Extract the three RGB colour channels")
#     R = img_mat[:,:,0]
#     G = img_mat[:,:,1]
#     B = img_mat[:,:,2]
#     # Step 2 - Combine the three colour channels by flatten each channel 
# 	# then stacking the flattened channels together.
#     # This gives the "colour_samples"
#     R_flat = R.flatten()
#     G_flat = G.flatten()
#     B_flat = B.flatten()
#     R_flat = R_flat.reshape((R_flat.shape[0],1))
#     G_flat = G_flat.reshape((G_flat.shape[0],1))
#     B_flat = B_flat.reshape((B_flat.shape[0],1))
#     colour_samples = np.hstack((R_flat,G_flat,B_flat))
#     #print("step 2: Combine the three colour channels")
#     # Step 3 - Perform Meanshift  clustering
#     ms_clf = MeanShift(bin_seeding=True)
#     ms_labels = ms_clf.fit_predict(colour_samples)
#     #print("step 3: Perform Meanshift  clustering")
#     # Step 4 - reshape ms_labels back to the original image shape 
# 	# for displaying the segmentation output 
#     ms_labels_reshape = np.reshape(ms_labels,original_shape[:2])
#     cluster_centers = ms_clf.cluster_centers_
#     n_clusters_ = np.unique(ms_labels)
#     #########show clusters in 3D
#     # print("step 4: Drawing result")
#     # Axes3D = Axes3D
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')

#     # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#     # for k, col in zip(range(len(n_clusters_)), colors):
#     #     my_members = ms_labels == k
#     #     # print("k is ",k)
#     #     # print("my member is ",my_members)
#     #     # print(colour_samples)
#     #     cluster_center = cluster_centers[k]
#     #     ax.scatter(colour_samples[my_members, 0], colour_samples[my_members, 1], colour_samples[my_members,2],c = col)
#     #     ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2],marker='*', c='k', s=200, zorder=10)
#     # plt.title('Estimated number of clusters: %d' % len(n_clusters_))
#     # plt.show()
#     #########show clusters in 3D

#     #
#     # +--------------------+
#     # |     Question 2     |
#     # +--------------------+
#     #
#     # TODO: perform Watershed on image
#     # Follow the hints in the lab spec.

#     # Step 1 - Convert the image to gray scale
#     # and convert the image to a numpy matrix
#     gray = np.array(img.convert('L'))
#     # Step 2 - Calculate the distance transform
#     # Hint: use     ndi.distance_transform_edt(img_array)
#     distance = ndi.distance_transform_edt(gray)
#     #Find peaks in an image as coordinate list or boolean mask.
#     local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
#                             labels=gray)
#     # Step 3 - Generate the watershed markers
#     # Hint: use the peak_local_max() function from the skimage.feature library
#     # to get the local maximum values and then convert them to markers
#     # using ndi.label() -- note the markers are the 0th output to this function
#     markers = ndi.label(local_maxi)[0]
#     # Step 4 - Perform watershed and store the labels
#     # Hint: use the watershed() function from the skimage.morphology library
#     # with three inputs: -distance, markers and your image array as a mask
#     ws_labels = watershed(-distance,markers,mask = gray)
#     # Display the results
#     plot_three_images(img_path, img, "Original Image", ms_labels_reshape, "MeanShift Labels",
#                       ws_labels, "Watershed Labels")

# #     # If you want to visualise the watershed distance markers then try
# #     # plotting the code below.
# #     # plot_three_images(img_path, img, "Original Image", -distance, "Watershed Distance",
# #     #                   ws_labels, "Watershed Labels")
# #%%
# #
# +-------------------+
# |     Extension     |
# +-------------------+
#
# Loop for the extension component
for img_path in ext_images:
    img = Image.open(img_path)
    img.thumbnail(size)
    #print("extention###############")
    # meanshift
    img_mat = np.array(img)[:, :, :3]
    original_shape = img_mat.shape
    #print("step 1: Extract the three RGB colour channels")
    R = img_mat[:,:,0]
    G = img_mat[:,:,1]
    B = img_mat[:,:,2]
    R_flat = R.flatten()
    G_flat = G.flatten()
    B_flat = B.flatten()
    R_flat = R_flat.reshape((R_flat.shape[0],1))
    G_flat = G_flat.reshape((G_flat.shape[0],1))
    B_flat = B_flat.reshape((B_flat.shape[0],1))
    colour_samples = np.hstack((R_flat,G_flat,B_flat))
    #print("step 2: Combine the three colour channels")
    ms_clf = MeanShift(bin_seeding=True)
    ms_labels = ms_clf.fit_predict(colour_samples)
    #print("step 3: Perform Meanshift  clustering")
    ms_labels_reshape = np.reshape(ms_labels,original_shape[:2])

    #watershed
    #print("step 4: Perform watershed")
    gray = np.array(img.convert('L'))
    distance = ndi.distance_transform_edt(gray)
    shapes = distance.shape
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=gray)#np.ones(shapes)
    markers = ndi.label(local_maxi)[0]
    ws_labels = watershed(-distance,markers,mask = gray)


    # # Display the results
    plot_three_images(img_path, img, "Original Image", ms_labels_reshape, "MeanShift Labels",
                      ws_labels, "Watershed Labels")

    ########CV2 watershed
    # # TODO: perform an optimisation and then watershed on image
    # img1 = cv2.imread(img_path)
    # gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) 

    # ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # # noise removal 腐蚀
    # kernel = np.ones((3,3),np.uint8)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # # sure background area 膨胀
    # sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # #cv2.imshow('Background Image', sure_bg)

    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # # Finding unknown region 边界
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg,sure_fg)
    # # cv2.imshow('Foreground Image', sure_fg)
    # # cv2.imshow('Unknown', unknown)
    # # Marker labelling mark背景：0，其他的用别的整数表示
    # ret, markers = cv2.connectedComponents(sure_fg)
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers+1
    # # Now, mark the region of unknown with zero 这里unknown中的白色的环状区域为未知区域
    # markers[unknown==255] = 0
    # #marker32 = np.int32(marker)
    # markers = cv2.watershed(img1,markers)
    # m = cv2.convertScaleAbs(markers)

    # colors = ['red','green','blue','purple','red','green','blue','purple','red','green','blue','purple']
    # print(np.unique(markers))
    # fig = plt.figure(figsize=(8,8))
    # plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))

    # cv2.imshow("markers",m)
    # cv2.waitKey(0)
    # img1[markers == -1] = [255,0,0]
    ########CV2 watershed


















