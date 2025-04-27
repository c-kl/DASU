from PIL import Image
import numpy as np
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from glob import glob
import math
import cv2
import os
from six.moves import xrange
from PIL import Image
import ssim_matlab
#Please use grayscale images. In utils.py, plt.imsave() save the images as pseudo-color image,
# while cv2.imwrite() save the images as grayscale image. Please note normalization.
dataset_pre = '.' #Please write the folder path containing the predicted results here
dataset_gt = '.' #Please write the folder path containing the ground truth (GT) results here

data1 = sorted(glob(os.path.join(
    dataset_pre, "*.png")))
data2 = sorted(glob(os.path.join(
    dataset_gt, "*.png")))

psnr_list = []
ssim_list = []
ssim_list_sk = []
psnr_list_sk = []


# psnr2_list = []
for filename in data1:
    for filename2 in data2:

        i = filename.split('/')[-1].split('.')[0]
        j = filename2.split('/')[-1].split('.')[0]

        if i == j:
            print(i)
            img_pre = Image.open(filename)
            img_gt = Image.open(filename2)
            img_pre = cv2.imread(filename, 0)
            img_gt = cv2.imread(filename2, 0)

            ssim_list_sk.append(compare_ssim(img_pre, img_gt))
            psnr_list_sk.append(compare_psnr(img_pre, img_gt))
            ###############ssim

            img_pre = img_pre.astype(np.double)
            img_gt = img_gt.astype(np.double)
            diff = img_gt - img_pre
            h, w = diff.shape
            r = np.sqrt(np.sum(np.power(diff, 2) / (h * w)))
            psnr = 20 * math.log10(255.0 / r)
            ssim_Matlab = ssim_matlab.ssim(img_gt, img_pre)
            psnr_list.append(psnr)
            ssim_list.append(ssim_Matlab)

print('psnr: ', np.mean(psnr_list))
print('psnr_sk: ', np.mean(psnr_list_sk))
print('ssim: ', np.mean(ssim_list))
print('ssim_sk: ', np.mean(ssim_list_sk))

