import cv2
import numpy as np
import os

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import cv2 as cv

'''
    Generate superpixel-based pseudolabels: SLIC and SEEDS-based methods
'''

def slic_():
    path_ = './2d_nontarget_image'
    dirs_img = os.listdir(path_)
    imgs_ = []
    count = 0
    for i in range(len(dirs_img)):
        # for i in range(1):
        image = img_as_float(cv2.imread(os.path.join(path_, dirs_img[i])))
        count +=1
        segments = slic(image, n_segments=200, max_iter=90, sigma=5)
        cv2.imwrite(os.path.join('./slic_labels/', 'super_slic'+dirs_img[i][5:]), segments)
    print(count)

# slic_()

def seeds_():
    path_ = './2d_nontarget_image'
    dirs_img = os.listdir(path_)
    imgs_ = []
    count = 0
    for i in range(len(dirs_img)):
        # for i in range(1):
        img = cv2.imread(os.path.join(path_, dirs_img[i]))
        count += 1

        seeds = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1],img.shape[0],img.shape[2],399,30,3,5,True) # Initialize the seeds item
        seeds.iterate(img,90)
        label_seeds = seeds.getLabels()
        # mask_seeds = seeds.getLabelContourMask()
        # number_seeds = seeds.getNumberOfSuperpixels()
        # mask_inv_seeds = cv2.bitwise_not(mask_seeds)
        # img_seeds = cv2.bitwise_and(img,img,mask =  mask_inv_seeds)
        # cv2.imshow("img_seeds",img_seeds)
        # cv2.imwrite(os.path.join('./super_seeds_mask.png'), mask_seeds)
        # cv2.imwrite(os.path.join('./super_seeds_img.png'), img_seeds)
        cv2.imwrite(os.path.join('./seeds_labels/', 'super_seed'+dirs_img[i][5:]), label_seeds)
    print(count)
# seeds_()

def check_sp():
    path_ = './slic_labels'
    dirs_img = os.listdir(path_)
    imgs_ = []
    count = 0
    for i in range(len(dirs_img)):
        lb = cv2.imread(os.path.join(path_, dirs_img[i]), 0)
        lb_ = np.unique(lb)
        print("{}: {}".format(dirs_img[i], lb_))

# check_sp()
