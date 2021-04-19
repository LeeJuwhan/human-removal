import os
import cv2
import numpy as np
from time import time
import natsort



def stackImagesKeypointMatching(PATH):

    orb = cv2.ORB_create()
    # orb = cv2.SIFT()
    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)
    
    file_list = os.listdir(PATH)
    file_list = natsort.natsorted(file_list)

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    i = 0
    for file in file_list: 
        print(file)
        
        image = cv2.imread(PATH + file,cv2.IMREAD_UNCHANGED )                
        imageF = image.astype(np.float32)
        
        h,w,c = image.shape          
        print(file)
        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)
        image_mask = np.zeros((h,w,c))        
        
        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des            
            
        else:
            
             # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32(
                [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            w, h, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (h, w))      
            
            
            image_mask = np.where(stacked_image == 0, imageF, image_mask)
            
            stacked_image = image_mask + stacked_image            
            
        i = i+1
    stacked_image = (stacked_image).astype(np.uint8)
    return stacked_image