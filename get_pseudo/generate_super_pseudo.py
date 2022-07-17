import matplotlib.pyplot as plt
import copy
import skimage

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.measure import label
import scipy.ndimage.morphology as snm
from skimage import io
import argparse
import numpy as np
import glob
import cv2

import SimpleITK as sitk
import os

'''
    Generate superpixel-based pseudolabels ---- 2d images version
'''

to01 = lambda x: (x - x.min()) / (x.max() - x.min())


def superpix_pxl(img, method='fezlen', **kwargs):

    if method == 'fezlen':
        seg_func = skimage.segmentation.felzenszwalb
    else:
        raise NotImplementedError

    seg = seg_func(img, min_size=400, sigma=1) # 400 MIDDLE
    # seg = seg_func(img, min_size=100, sigma=1) # 100 SMALLE

    return seg

# thresholding the intensity values to get a binary mask
def fg_mask2d(img_2d, thresh):  # change this by your need
    mask_map = np.float32(img_2d > thresh)

    def getLargestCC(segmentation):  # largest connected components
        labels = label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC

    if mask_map.max() < 0.999:
        return mask_map
    else:
        post_mask = getLargestCC(mask_map)
        fill_mask = snm.binary_fill_holes(post_mask)
        fill_mask = fill_mask.astype(np.uint8)
    return fill_mask


# remove superpixels within the empty regions
def superpix_masking(raw_seg2d, mask2d):
    raw_seg2d = np.int32(raw_seg2d)
    lbvs = np.unique(raw_seg2d)
    max_lb = lbvs.max()
    raw_seg2d[raw_seg2d == 0] = max_lb + 1
    lbvs = list(lbvs)
    lbvs.append(max_lb)
    raw_seg2d = raw_seg2d * mask2d
    lb_new = 1
    out_seg2d = np.zeros(raw_seg2d.shape)
    for lbv in lbvs:
        if lbv == 0:
            continue
        else:
            out_seg2d[raw_seg2d == lbv] = lb_new
            lb_new += 1

    return out_seg2d


def superpix_wrapper(img, verbose=False, fg_thresh=1e-4):
    raw_seg_2d = superpix_pxl(img) # modified to 2d

    _fgm = fg_mask2d(img, fg_thresh)
    _out_seg = superpix_masking(raw_seg_2d, _fgm)

    return _fgm, _out_seg

# Generate pseudolabels for every image and save them
path_ = './2d_nontarget_image'

dirs_img = os.listdir(path_)
imgs_ = []
count = 0
for i in range(len(dirs_img)):
    img = cv2.imread(os.path.join(path_, dirs_img[i]), 0)

    count += 1
    out_fg, out_seg = superpix_wrapper(img, fg_thresh=1e-4 + 50) # fg_thresh value can be modified
    cv2.imwrite(os.path.join('', 'superpixel'+dirs_img[i][5:]), out_seg)

print(count)
