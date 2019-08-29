import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys

def read_img_from_csv(filename):
    if filename is None:
        sys.exit()
    im = pd.read_csv(filename, header=None, dtype=np.float32)
    #######################extract labels######################
    im_label_df = im.iloc[:,-1]
    im_label_value = []
    for i in im_label_df.values:
        if i not in im_label_value:
            im_label_value.append(i)
    ############################################################
    im = im.iloc[:,:-1]#last column is the label
    total_height = im.shape[0]
    imgheight = im.shape[1]  
    n_imgs = total_height//imgheight

    imgs = []
    data = list(im.values)
    for i in range(n_imgs):
        imgs.append(data[i*imgheight : (i+1)*imgheight])

    my_dpi = 192
    for i in range(n_imgs):
        
        fig = plt.figure(figsize=(512/my_dpi, 512/my_dpi), dpi=my_dpi)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        print(len(imgs[i]))
        ax.imshow(imgs[i], cmap = 'gray')
        plt.savefig("./test_output/" + str(im_label_value[i]) + ".png",format='png',dpi = my_dpi)
        plt.imsave()

        # plt.figure(i)
        # plt.imshow(imgs[i], 'gray')

    #plt.show()

if __name__ == "__main__":
    
    fn = 'pixel_data_new.csv'
    # imgsize = 512
    read_img_from_csv(fn)