# Template for computing SIFT features
import cv2
import numpy as np
from matplotlib import pyplot as plt


class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]=0
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.04
            params["edge_threshold"]=10
            params["sigma"]=1.6

        detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=params["n_features"],
                nOctaveLayers=params["n_octave_layers"],
                contrastThreshold=params["contrast_threshold"],
                edgeThreshold=params["edge_threshold"],
                sigma=params["sigma"])

        return detector

class rotator():
    def __init__(self,image,angle,params=None):
        self.image = image
        self.angle = angle
        self.rotated_img = self.rotate(self.image,self.angle,params)

    def rotate(self,image, angle, params=None):
        # 获取图像尺寸
        (h, w) = image.shape[:2]
        # 若未指定旋转中心，则将图像中心设为旋转中心
        if(params is None):
            center = (w / 2, h / 2)
            scale = 1
        else:
            center = params["center"]
            scale = params["scale"]

        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
     
        # 返回旋转后的图像
        return rotated



if __name__ == '__main__':
    # 1. Read the colour image
    img = cv2.imread('NotreDame.jpg')
    # For task 2 only, rotate the image by 45 degrees
    rotator = rotator(img,-45)
    img_rotate = rotator.rotated_img

    # 2. Convert image to greyscale	
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray2= cv2.cvtColor(img_rotate,cv2.COLOR_BGR2GRAY)
    
    # 3. Initialise SIFT detector (with varying parameters)
    sift = SiftDetector().get_detector(None)
    
    # 4. Detect and compute SIFT features
    #kp = sift.detect(gray,None)
    

    # 5. Visualise detected keypoints on the colour image 
    # img_1=cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('sift_keypoints.jpg',img_1)

    # 6. compute sift key points and descriptor
    # kp, des = sift.detectAndCompute(gray,None)
    # print("number of descriptor is: ",len(des))
    # print("descriptor values are: \n", des)
    # to reduce the number of key point to 1/4
    # n_keypoints = len(des)//4
    # print("1/4 is ",n_keypoints)
    sift = SiftDetector().get_detector({"n_features":0,
            "n_octave_layers":3,
            "contrast_threshold":0.1,
            "edge_threshold":10,
            "sigma":1.6})
    kp1, des1 = sift.detectAndCompute(gray,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)
    print("kp1 number is ", len(kp1))
    print("kp2 number is ", len(kp2))

    cv2.drawKeypoints(img,kp1,img)#,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(img_rotate,kp2,img_rotate)#,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints_less.jpg',img)
    cv2.imwrite('sift_keypoints_less_part2.jpg',img_rotate)


    #extra......1
    for angle in range(0,-225,-45):

        img_rotate = rotator.rotate(img, angle)

        gray2= cv2.cvtColor(img_rotate,cv2.COLOR_BGR2GRAY)
        sift = SiftDetector().get_detector({"n_features":0,
                "n_octave_layers":3,
                "contrast_threshold":0.1,
                "edge_threshold":8,
                "sigma":1.6})
        kp, des = sift.detectAndCompute(gray2,None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        img3 = cv2.drawMatches(img,kp1,img_rotate,kp,matches,None,flags=2)

        plt.imshow(img3),plt.show()
     #extra......2
    (h, w) = img.shape[:2]
    img_upscale = rotator.rotate(img, 0,{"center":(w / 2, h / 2), "scale":4})
    gray3= cv2.cvtColor(img_upscale,cv2.COLOR_BGR2GRAY)
    sift = SiftDetector().get_detector({"n_features":0,
        "n_octave_layers":3,
        "contrast_threshold":0.1,
        "edge_threshold":8,
        "sigma":1.6})
    kp3, des3 = sift.detectAndCompute(gray3,None)
    cv2.drawKeypoints(img_upscale,kp3,img_upscale)
    cv2.imwrite('upscale.jpg',img_upscale)
    #matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des3, k=2)
    #apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    result = cv2.drawMatchesKnn(img,kp1,img_upscale,kp3,good,None,flags=2)
    plt.imshow(result),plt.show()

        # cv2.drawKeypoints(img_rotate,kp1,img_rotate)
        # cv2.imwrite('degree {0}.jpg'.format(angle),img_rotate)
























