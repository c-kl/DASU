import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import glob
import random
from PIL import Image
import tqdm

from utils import make_coord, visualize_2d


def to_pixel_samples(depth):
    """ Convert the image to coord-RGB pairs.
        depth: Tensor, (1, H, W)
    """
    coord = make_coord(depth.shape[-2:], flatten=True).view(depth.shape[-2], depth.shape[-1], 2)  # [H，W, 2]
    pixel = depth.view(-1, 1)  # [H*W, 1]
    return coord, pixel


class Geo_MiddleburyDataset(Dataset):
    def __init__(self, root, split='test', scale=8, augment=True, downsample='bicubic', pre_upsample=False,
                 to_pixel=False, sample_q=None, input_size=None, noisy=False):
        super().__init__()
        self.root = root
        self.split = split
        self.scale = scale
        self.augment = augment
        self.downsample = downsample
        self.pre_upsample = pre_upsample
        self.to_pixel = to_pixel
        self.sample_q = sample_q
        self.input_size = input_size
        self.noisy = noisy

        if self.split == 'train':
            raise AttributeError('Middlebury dataset only support test mode.')
        else:
            self.image_files = sorted(glob.glob(os.path.join(root, '*output_color*')))
            self.depth_files = sorted(glob.glob(os.path.join(root, '*output_depth*')))
            self.HF_files = sorted(glob.glob(os.path.join(root, '*output_HF*')))
            assert len(self.image_files) == len(self.depth_files)
            self.size = len(self.image_files)

        print("========  DFS*** Use middlebury datalader =======")

    def __getitem__(self, idx):

        image_file = self.image_files[idx]
        depth_file = self.depth_files[idx]
        HF_file = self.HF_files[idx]
        image = cv2.imread(image_file).astype(np.uint8)  # [H, W, 3]

        depth_hr = cv2.imread(depth_file)[:, :, 0].astype(np.float32)  # [H, W]
        HF_hr = cv2.imread(HF_file)[:, :, 0].astype(np.float32)  # [H, W]
        depth_min = depth_hr.min()
        depth_max = depth_hr.max()
        depth_hr = (depth_hr - depth_min) / (depth_max - depth_min)

        lpls_min = HF_hr.min()
        lpls_max = HF_hr.max()
        HF_hr = (HF_hr - lpls_min) / (lpls_max - lpls_min)

        # print('image --', image.shape)
        # print('HR', depth_hr.shape)

        # crop to make divisible
        # print("---------------->", self.scale)
        h, w = image.shape[:2]
        h = h - int(h % self.scale)
        w = w - int(w % self.scale)
        image = image[:h, :w]
        depth_hr = depth_hr[:h, :w]
        HF_hr = HF_hr[:h, :w]

        # crop after rescale
        if self.input_size is not None:
            print("crop")
            x0 = random.randint(0, image.shape[0] - self.input_size)
            y0 = random.randint(0, image.shape[1] - self.input_size)
            image = image[x0:x0 + self.input_size, y0:y0 + self.input_size]
            depth_hr = depth_hr[x0:x0 + self.input_size, y0:y0 + self.input_size]
            HF_hr = HF_hr[x0:x0 + self.input_size, y0:y0 + self.input_size]

        h, w = image.shape[:2]

        if self.downsample == 'bicubic':
            depth_lr = np.array(Image.fromarray(depth_hr).resize((int(w // self.scale), int(h // self.scale)),
                                                                 Image.BICUBIC))  # bicubic, RMSE=7.13
            HF_lr = np.array(Image.fromarray(HF_hr).resize((int(w // self.scale), int(h // self.scale)),
                                                                 Image.BICUBIC))  # bicubic, RMSE=7.13
            image_lr = np.array(Image.fromarray(image).resize((int(w // self.scale), int(h // self.scale)),
                                                              Image.BICUBIC))  # bicubic, RMSE=7.13
            # depth_lr = np.array(Image.fromarray(depth_hr).resize((w//self.scale, h//self.scale), Image.BICUBIC))
            # image_lr = np.array(Image.fromarray(image).resize((w//self.scale, h//self.scale), Image.BICUBIC))
        elif self.downsample == 'nearest-right-bottom':
            depth_lr = depth_hr[(self.scale - 1)::self.scale, (self.scale - 1)::self.scale]
            image_lr = image[(self.scale - 1)::self.scale, (self.scale - 1)::self.scale]
        elif self.downsample == 'nearest-center':
            depth_lr = np.array(Image.fromarray(depth_hr).resize((w // self.scale, h // self.scale), Image.NEAREST))
            image_lr = np.array(Image.fromarray(image).resize((w // self.scale, h // self.scale), Image.NEAREST))
        elif self.downsample == 'nearest-left-top':
            depth_lr = depth_hr[::self.scale, ::self.scale]
            image_lr = image[::self.scale, ::self.scale]
        else:
            raise NotImplementedError

        image = image.astype(np.float32).transpose(2, 0, 1) / 255
        image_lr = image_lr.astype(np.float32).transpose(2, 0, 1) / 255  # [3, H, W]

        image = (image - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3,
                                                                                                                     1,
                                                                                                                     1)
        image_lr = (image_lr - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array(
            [0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # follow DKN, use bicubic upsampling of PIL
        depth_lr_up = np.array(Image.fromarray(depth_lr).resize((w, h), Image.BICUBIC))
        HF_lr_up = np.array(Image.fromarray(HF_lr).resize((w, h), Image.BICUBIC))

        if self.pre_upsample:
            depth_lr = depth_lr_up
            HF_lr = HF_lr_up

        # to tensor
        image = torch.from_numpy(image).float()
        image_lr = torch.from_numpy(image_lr).float()
        depth_hr = torch.from_numpy(depth_hr).unsqueeze(0).float()
        depth_lr = torch.from_numpy(depth_lr).unsqueeze(0).float()
        depth_lr_up = torch.from_numpy(depth_lr_up).unsqueeze(0).float()
        HF_hr = torch.from_numpy(HF_hr).unsqueeze(0).float()
        HF_lr = torch.from_numpy(HF_lr).unsqueeze(0).float()
        HF_lr_up = torch.from_numpy(HF_lr_up).unsqueeze(0).float()
        # transform
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            image = augment(image)
            image_lr = augment(image_lr)
            depth_hr = augment(depth_hr)
            depth_lr = augment(depth_lr)
            depth_lr_up = augment(depth_lr_up)
            HF_hr = augment(HF_hr)
            HF_lr = augment(HF_lr)
            HF_lr_up = augment(HF_lr_up)

        image = image.contiguous()
        image_lr = image_lr.contiguous()
        depth_hr = depth_hr.contiguous()
        depth_lr = depth_lr.contiguous()
        depth_lr_up = depth_lr_up.contiguous()
        HF_hr = HF_hr.contiguous()
        HF_lr = HF_lr.contiguous()
        HF_lr_up = HF_lr_up.contiguous()
        LF_hr = depth_hr - HF_hr
        LF_lr = depth_lr - HF_lr
        LF_lr_up  = depth_lr_up - HF_lr_up

        # to pixel
        if self.to_pixel:
            hr_coord, hr_pixel = to_pixel_samples(depth_hr)

            lr_distance_h = 2 / depth_lr.shape[-2]
            lr_distance_w = 2 / depth_lr.shape[-1]
            lr_distance = torch.tensor([lr_distance_h, lr_distance_w])
            field = torch.ones([8])
            cH, cW, _ = hr_coord.shape
            ch = cH // 2
            cw = cW // 2

            f1 = abs(hr_coord[ch + 1, cw - 1] - hr_coord[ch, cw])
            field[0:2] = f1 / lr_distance
            f2 = abs(hr_coord[ch - 1, cw - 1] - hr_coord[ch, cw])
            field[2:4] = f2 / lr_distance
            f3 = abs(hr_coord[ch + 1, cw + 1] - hr_coord[ch, cw])
            field[4:6] = f3 / lr_distance
            f4 = abs(hr_coord[ch - 1, cw + 1] - hr_coord[ch, cw])
            field[6:] = f4 / lr_distance

            return {
                'hr_image': image,
                'lr_image': image_lr,
                'lr_depth': depth_lr,
                'hr_depth': depth_hr,
                'lr_depth_up': depth_lr_up,

                'hr_coord': hr_coord,
                'min': depth_min,
                'max': depth_max,
                'idx': idx,
                'field': field,

                'depth_LF_lr': LF_lr,
                'depth_LF_hr': LF_hr,
                'depth_LF_lr_up': LF_lr_up,

                'depth_HF_hr': HF_hr,
                'depth_HF_lr': HF_lr,
                'depth_HF_lr_up': HF_lr_up,
            }

    def __len__(self):
        return self.size
