import os
import matplotlib.pyplot as plt
import time
import cv2
import torch.nn.functional as F
import numpy as np
savepath=r'features_vis'
if not os.path.exists(savepath):
    os.mkdir(savepath)

# method 1
def draw_features(width, height, x, savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    # cmap = 'nipy_spectral'
    # plt.imshow(img[:, :, ::-1], cmap=plt.get_cmap(cmap))
    for i in range(width*height):
    # for i in range(1):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255
        img=img.astype(np.uint8)
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET)
        # img=cv2.applyColorMap(img, cv2.COLORMAP_HOT)
        img = img[:, :, ::-1]
        plt.imshow(img)
        # print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))

# method 2
def draw(output, save_name):
    output = output.mean(dim=1)
    output_np = output.data.cpu().numpy()
    mask = output_np.squeeze(0)
    norm_img = np.zeros(mask.shape)
    norm_img = cv2.normalize(mask, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(1-norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    # heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(savepath, save_name + '.png'), heat_img)