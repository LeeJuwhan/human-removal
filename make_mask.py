import cv2
import numpy as np

import test
import os
from os.path import basename
import natsort
import torch


os.environ['MKL_THREADING_LAYER'] = 'GNU'
def mask(file, input_path,mask_path):
    
        
    image = cv2.imread(input_path+file,cv2.IMREAD_UNCHANGED)
    #??
    image = cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
    kernel = np.ones((3,3),np.uint8)
    w,h,c = image.shape
    print(c)
    mask = np.zeros((w,h),np.uint8)
    
    mask2 = np.zeros((w,h),np.uint8)
    image2 = image.copy()
    
    mask = np.where(image[:,:,3] == 0, 255, mask)
    mask = cv2.dilate(mask,kernel, iterations =1)
    for i in range(c):
        image2[:,:,i] = np.where(image2[:,:,3] == 0,255,image2[:,:,i])        
    mask2 = np.where(image2[:,:,3] != 255, 255,mask2)    
    
    cv2.imwrite(mask_path + file,mask)    
    
    return mask2
    
def fill_zero(file,input_path,mask_path,mask2):
# inpaint and 
    image = cv2.imread(input_path+file,-1)
    #??
    image = cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
    _,_,channel = image.shape
    for c in range(channel) :        
        image[:,:,c] = np.where(image[:,:,3] != 255, 255,image[:,:,c])    
    image = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)    
    image = cv2.inpaint(image,mask2,3,cv2.INPAINT_TELEA)        
    cv2.imwrite(input_path+file,image)
    
    
def imresize () :
    for fname in file_list:    
        path_ori = path + fname    
        img = cv2.imread(path_ori,cv2.IMREAD_COLOR)
        height = img.shape[0]
        width = img.shape[1]
        c_fname = fname    
        path_conv = result_path + c_fname        
        c_img = cv2.resize(img,dsize = (256,256), interpolation=cv2.INTER_AREA)        
        print(c_fname)
        cv2.imwrite(path_conv,c_img)

def find_mask_roi3(file, input_path,mask_path,mask_roi_path, image_roi_path):
    #512 * 512
    image = cv2.imread(input_path + file,1)
    mask = cv2.imread(mask_path + file,0)    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cx = []
    cy = []
    cx = [0] * len(contours)
    cy = [0] * len(contours)
    idx = []
    idx = [0] * len(contours)
    print(len(contours))
    h,w,c = image.shape
    file_idx = 0
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)        
        if M['m00'] == 0:
            idx[i] = False
            continue
        # print(M)
        
        idx[i] = True
        file_idx = file_idx + 1
        cx[i] = int(M['m10']/M['m00'])        
        cy[i] = int(M['m01']/M['m00'])
        
        if cx[i] + 256 > w:
            cx[i] = abs((((cx[i] + 256) - w) - cx[i]))
        if cx[i] - 256 <0 :
            cx[i] = abs(cx[i] - 256) + cx[i]
        if cy[i] + 256 > h:
            cy[i] = abs((((cy[i] + 256) - h) - cy[i]))
        if cy[i] - 256 < 0:
            cy[i] = abs(cy[i] - 256) + cy[i]
        
        mask_roi = mask[cy[i] - 256 : cy[i] + 256, cx[i] - 256 : cx[i] + 256]
        image_roi = image[cy[i] - 256 : cy[i] + 256, cx[i] - 256 : cx[i] + 256]
                
        cv2.imwrite(mask_roi_path + '%d.png'%i,mask_roi)
        cv2.imwrite(image_roi_path + '%d.png'%i, image_roi)
        
    return cx,cy, idx

def morphology(file):
    image = cv2.imread('aaaaa.png',-1)
    kernel = np.ones((3,3), np.uint8)    
    image = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)    
    cv2.imwrite('bbbbb.png', image)

def image_merge3(cx,cy,results_path,results_list,input_file,idx):
    torch.cuda.empty_cache()
    image = cv2.imread(input_file,1)
    results_idx = 0    
    print(results_path)
    
    for i in range(len(idx)):        
        if idx[i] == False:
            continue
        else :               
            results = cv2.imread(results_path  + results_list[results_idx],1)                                          
            image[cy[i] - 256 : cy[i] + 256, cx[i] - 256 : cx[i] + 256] = results    
            results_idx = results_idx + 1        
    return image

def image_merge4(cx,cy,results_path,results_list,input_file,idx,MM):
    #특정 개수 256 --> 512
    image = cv2.imread(input_file,1)
    results_idx = 0    
    print(results_path)
    
    for i in range(len(idx)):        
        if idx[i] == False:
            continue
        else :                 
            if MM[i] == True:
                results = cv2.imread(results_path + results_list[results_idx],1)                  
                image[cy[i] - 128 : cy[i] + 128, cx[i] - 128 : cx[i] + 128] = results    
                results_idx = results_idx + 1        
            else :
                results = cv2.imread(results_path  + results_list[results_idx],1)                              
                results = cv2.resize(results,dsize=(512,512),interpolation = cv2.INTER_AREA)
                image[cy[i] - 256 : cy[i] + 256, cx[i] - 256 : cx[i] + 256] = results    
                results_idx = results_idx + 1        
    return image

def line_inpainting(image, results_path,mask2):
    # image = cv2.imread('aaaaa.png',-1)
    w,h,c = image.shape
    mask = np.zeros((w,h,c),np.uint8)
    # mask2 = np.zeros((w,h),np.uint8)
    mask = np.where(image == 255, 255, mask)    
    # mask2 = np.where(mask[:,:,0] == 255, 255, mask2)    
    image = cv2.inpaint(image,mask2,3,cv2.INPAINT_TELEA)    
    cv2.imwrite(results_path + 'result.png',image)
    
def main(input_path, mask_path, mask_roi_path, image_roi_path,results_path):
    
    file_list = os.listdir(input_path)
    for file in file_list:                
        mask2 = mask(file,input_path,mask_path)
        fill_zero(file,input_path,mask_path,mask2)
        
    mask_list = os.listdir(mask_path)
    for file in mask_list:        
        cx,cy,idx = find_mask_roi3(file,input_path,mask_path, mask_roi_path, image_roi_path)


    
    #### inpainting : edge-connect
    test.inpainting(mask_roi_path, image_roi_path,results_path)
    ####end
    results_list = os.listdir(results_path)
    results_list = natsort.natsorted(results_list)
    input_file = input_path + file_list[0]
    

    image = image_merge3(cx,cy,results_path,results_list,input_file,idx)    

    
    cv2.imwrite(results_path +  "results.png",image)

if __name__ == "__main__":    
    main(input_path,mask_path, mask_roi_path, image_roi_path,results_path)
    
    
    