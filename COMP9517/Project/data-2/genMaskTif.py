
import tifffile
import os,sys
import numpy as np
def gen3Dtif(inputMask,inputImage,outputMask,outputImage):
    # inputMask = "/Users/congcong/Desktop/COMP9517/Project/data-2/labels_tiff"
    # inputImage = "/Users/congcong/Desktop/COMP9517/Project/data-2/output"
    #define hyper-parameters
    maskPaths = [inputMask + f'/train-labels{i:d}.tif' for i in range(30)]
    imagePaths = [inputImage + f'/{i:d}_predict.tif' for i in range(30)]
    allInput = [maskPaths,imagePaths]
    allOutput = [outputMask,outputImage]
    for i in range(len(allInput)):
        stacked = []
        #produce stacked tif images
        for imgs in allInput[i]:
            print(imgs)
            if(i == 0):
                maskImg = tifffile.imread(inputMask+"/"+imgs)
            else:
                maskImg = tifffile.imread(inputImage+"/"+imgs)

            maskImg = np.expand_dims(maskImg,axis=0)
            if(len(stacked)==0):
                stacked = maskImg
            else:
                stacked = np.vstack((stacked,maskImg))
        #output tacked tif image
        if not os.path.exists(allOutput[i]):
            os.makedirs(allOutput[i])
        if(i == 0):
            tifffile.imsave(allOutput[i] + '/allMask.tif', stacked)
        else:
            tifffile.imsave(allOutput[i] + '/allImage.tif', stacked)

if __name__ == "__main__":
    #params: 1.input mask path 2. input image path 3. output mask path 4. output image path
    gen3Dtif(sys.argy[1],sys.argy[2],sys.argy[3],sys.argy[4])



