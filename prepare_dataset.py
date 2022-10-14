# -*- coding: utf-8 -*-
"""
@author: Heet Sakaria
"""

from logging import root
from pathlib import Path
import numpy as np;
import os;
import cv2;
from glob import glob;
import scipy.io
from progress_timer import progress_timer;

def set_name(path):
    index_list = glob(path + '\INDEX\*.mat')
    name, count = [], [] 
    
    for i in range(len(index_list)):
        index_mat = (scipy.io.loadmat(index_list[i],  squeeze_me=True, struct_as_record = False))
        name.append(index_mat['index'].set_name)
        count.append((index_mat['index'].cc).size + (index_mat['index'].img).size)
    
    return (name, count)
        
def check_set(path):
    index_list = glob(path + '\\real_illum\*.mat');
    name, count = set_name(path)
    i, j = 0, 0
    name_out, count_out = [], []
    
    while i < len(index_list):
        if name[j] in index_list[i]:
            name_out.append(name[j])
            count_out.append(count[j])
            j += 1
            i += 1
            
        else:
            j += 1
            
    return (name_out, count_out)
    

def convert_to_8bit(arr, clip_percentile):
    """
    This method is derived from work of https://github.com/WhuEven/CNN_model_ColorConstancy

    This method converts 12-bit image to 8-bit channel.

    Args:
        arr (numpy.narray): 12-bit image mat/numpy array
        clip_percentile
    """    

    arr = np.clip(arr * (255.0 / np.percentile(arr, 100 - clip_percentile, keepdims=True)), 0, 255)
    return arr.astype(np.uint8)


def white_balance_image(img12, bit, Gain_R, Gain_G, Gain_B):
    """
    This method is derived from work of https://github.com/WhuEven/CNN_model_ColorConstancy

    This method converts 12-bit image to 8-bit channel and performs white balancing.

    Args:
        img12 (numpy.narray): image mat/numpy array
        bit (int): bits of image
        Gain_R: illumination value of Red channel
        Gain_G: illumination value of Green channel
        Gain_B: illumination value of Blue channel
    """    
    img12 = (img12/(2**bit - 1))*100.0; #12bit data
    image = convert_to_8bit(img12, 2.5)
    image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
    image[:, :, 0] = np.minimum(image[:, :, 0] * Gain_R,255)
    image[:, :, 1] = np.minimum(image[:, :, 1] * Gain_G,255)
    image[:, :, 2] = np.minimum(image[:, :, 2] * Gain_B,255)
            
    gamma = 1/2.2
    image = pow(image, gamma) * (255.0/pow(255,gamma))
    image = np.array(image,dtype=np.uint8)
    image8 = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    return image8
    
        
def WhiteBalanceDataSet(path, bit, Dataset):
    """
    This method is derived from work of https://github.com/WhuEven/CNN_model_ColorConstancy

    This method creates the ground truth images for model training.

    Args:
        path (str): path of root directory of dataset
        bit (int): bits of image
        Dataset (str): Type of Dataset(in our implementation we are using Shi_Gehler)
    """
    if Dataset == 'Shi_Gehler':
        path_gt = path + '\gt'
        print(path)
        mat = (scipy.io.loadmat(path_gt + '\\real_illum_568.mat',  squeeze_me=True, struct_as_record = False))
        path_img = path + '\png1'
        flist = glob(path_img + '\*.png')
        
    elif Dataset == 'Funt_et_al':
        mat = mat_illum(path)
        name, _ = check_set(path); 
        path_hdr = path + '\hdr_cc';   
        flist = glob(path_hdr + '\*.hdr')
        
    elif Dataset == 'Canon_600D':
        mat = (scipy.io.loadmat(path + '\\Canon600D_gt.mat',  squeeze_me=True, struct_as_record = False))
        path_img = path + '\png1';   
        flist = glob(path_img + '\*.png')
    
    pt = progress_timer(n_iter = len(flist), description = 'Preparing Groundtruth Images :')
    
    for i in range(0, len(flist)):
                    
        if Dataset == 'Shi_Gehler':
            img12 = cv2.imread(flist[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            
            if i >= 86: # For the Canon 5D the black level is 129
                img12 = np.maximum(0.,img12-129.)
            
            Gain_R= float(np.max(mat['real_rgb'][i]))/float((mat['real_rgb'][i][0]))
            Gain_G= float(np.max(mat['real_rgb'][i]))/float((mat['real_rgb'][i][1]))
            Gain_B= float(np.max(mat['real_rgb'][i]))/float((mat['real_rgb'][i][2]))
            
            image8 = white_balance_image(img12, bit, Gain_R, Gain_G, Gain_B)
            save_dir = os.path.join(path,'GroundTruth') + "\%04d.png" % (i+1)
            cv2.imwrite(save_dir, image8)
            # i += 1
            
        elif Dataset == 'Funt_et_al':
            #k  = 0;
            gamma_tone = 2.2
            tonemap = cv2.createTonemap(gamma_tone)
            

            img12 = cv2.imread(flist[i], cv2.IMREAD_UNCHANGED);               
            avg_R = np.mean(mat[i][:, 0]); avg_G = np.mean(mat[i][:, 1]); avg_B = np.mean(mat[i][:, 2])
            Gain_R = float(np.max((avg_R, avg_G, avg_B))/float(avg_R))
            Gain_G = float(np.max((avg_R, avg_G, avg_B))/float(avg_G))
            Gain_B = float(np.max((avg_R, avg_G, avg_B))/float(avg_B))
                
            img12[:, :, 0] = np.minimum(img12[:, :, 0] * Gain_B,255)
            img12[:, :, 1] = np.minimum(img12[:, :, 1] * Gain_G,255)
            img12[:, :, 2] = np.minimum(img12[:, :, 2] * Gain_R,255)
            
            image = tonemap.process(img12)
            clip_percentile = 0.2
            
            if np.any(np.isnan(image)):
                image[np.isnan(image)] = 0
                
            image = np.clip(image * (255.0 / np.percentile(image, 100 - clip_percentile, keepdims=True)), 0, 255)
          
            img_name = name[i] +  "_GT_%01d.png" % (i+1)
            save_dir = path_hdr.replace('hdr_cc', 'GroundTruth\\') + img_name
            cv2.imwrite(save_dir, image);         

        elif Dataset == 'Canon_600D':
            img12 = cv2.imread(flist[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            
            img12 = np.maximum(0.,img12 - 2048.)
            
            Gain_R= float(np.max(mat['groundtruth_illuminants'][i]))/float((mat['groundtruth_illuminants'][i][0]))
            Gain_G= float(np.max(mat['groundtruth_illuminants'][i]))/float((mat['groundtruth_illuminants'][i][1]))
            Gain_B= float(np.max(mat['groundtruth_illuminants'][i]))/float((mat['groundtruth_illuminants'][i][2]))
            
            image8 = white_balance_image(img12, bit, Gain_R, Gain_G, Gain_B)
            save_dir = path_img.replace('png1', 'GroundTruth') + "\%04d.png" % (i+1)
            cv2.imwrite(save_dir, image8)
            i += 1
                
            
        pt.update()
        
    pt.finish()


# dataset_name = 'Shi_Gehler'
# root_dir = os.getcwd()
# print(root_dir)
# dataset_path = os.path.join(root_dir,'Dataset',dataset_name)
# print(dataset_path)
# # print(path)
# # #convert2png(path, 14, Dataset);
# WhiteBalanceDataSet(dataset_path, 12, dataset_name)