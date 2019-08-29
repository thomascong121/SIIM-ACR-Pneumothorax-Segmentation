import os
import numpy as np
import tifffile
tif_list = os.listdir('/Users/congcong/Desktop/COMP9517/Project/data-2/output')
for tif_file in tif_list:
    image = tifffile.imread('/Users/congcong/Desktop/COMP9517/Project/data-2/output/' + tif_file)
    print('image shape: ',image.shape)
    print('image contains: ',np.unique(image))
    print(image[np.where(image > 1)])
    print(image[np.where(image <0 )])
    print("=========================")
