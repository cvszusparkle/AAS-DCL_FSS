import os
import cv2
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def save_pickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)


def compute_stats(voxels):
    if len(voxels) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    median = np.median(voxels)
    mean = np.mean(voxels)
    sd = np.std(voxels)
    mn = np.min(voxels)
    mx = np.max(voxels)
    percentile_99_5 = np.percentile(voxels, 99.5)
    percentile_00_5 = np.percentile(voxels, 00.5)
    return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5


def load_data(img_path, mask_path=None):
    # img = cv2.imread(img_path, 0)
    img = cv2.imread(img_path)
    img = img.transpose((2, 0, 1))
    size = img.shape
    if mask_path is not None:
        mask = cv2.imread(mask_path, 0)
    else:
        mask = None
    return img, mask, size


def get_properties(args):
    file = os.path.join(args.dir_data, 'dataset.pkl')
    if args.dataset in ['CHAOST2', 'MAALC', 'PseudoClass']:
        data_type = "*.png"
    else:
        data_type = None

    img_file = glob(os.path.join(args.dir_img, data_type))
    img_file.sort()
    mask_file = glob(os.path.join(args.dir_mask, data_type))
    mask_file.sort()

    if not os.path.exists(file) and img_file is not None:
        results = OrderedDict()
        assert args.n_channels in [1, 3], "Please check the value of n_channels"
        all_size = []
        if args.n_channels == 1:
            w = []
            with tqdm(total=len(img_file), desc=f'', unit='img') as pbar:
                for img_path, mask_path in zip(img_file, mask_file):
                    pbar.update(1)
                    img_npy, mask_npy, size = load_data(img_path, mask_path)
                    all_size.append(size)
                    w += list(img_npy.flatten()[::5])
            median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = compute_stats(w)
            results['all_size'] = all_size
            results['median'] = median
            results['mean'] = mean
            results['sd'] = sd
            results['mn'] = mn
            results['mx'] = mx
            results['percentile_99_5'] = percentile_99_5
            results['percentile_00_5'] = percentile_00_5
        elif args.n_channels == 3:
            w1, w2, w3 = [], [], []
            with tqdm(total=len(img_file), desc=f'', unit='img') as pbar:
                for img_path, mask_path in zip(img_file, mask_file):
                    pbar.update(1)
                    img_npy, mask_npy, size = load_data(img_path, mask_path)
                    all_size.append(size)
                    w1 += list(img_npy[0].flatten())
                    w2 += list(img_npy[1].flatten())
                    w3 += list(img_npy[2].flatten())

            median_1, mean_1, sd_1, mn_1, mx_1, percentile_99_5_1, percentile_00_5_1 = compute_stats(w1)
            median_2, mean_2, sd_2, mn_2, mx_2, percentile_99_5_2, percentile_00_5_2 = compute_stats(w2)
            median_3, mean_3, sd_3, mn_3, mx_3, percentile_99_5_3, percentile_00_5_3 = compute_stats(w3)
            results['all_size'] = all_size
            results['median'] = [median_1, mean_2, median_3]
            results['mean'] = [mean_1, mean_2, mean_3]
            results['sd'] = [sd_1, sd_2, sd_3]
            results['mn'] = [mn_1, mn_2, mn_3]
            results['mx'] = [mx_1, mx_2, mx_3]
            results['percentile_99_5'] = [percentile_99_5_1, percentile_99_5_2, percentile_99_5_3]
            results['percentile_00_5'] = [percentile_00_5_1, percentile_00_5_2, percentile_00_5_3]
        save_pickle(results, file)
    elif os.path.exists(file):
        results = load_pickle(file)
    else:
        results = None
    return results

