import torch
from torch import nn
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

import hyperparameters


def Fro_LOSS(x, y):
    return torch.norm(x - y, 'fro')

class XRayDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.crop_size = hyperparameters.crop_size
        # find all file pairs
        high_paths = []
        low_paths = []
        for filename in os.listdir(path):
            if "high" in filename:
                high_paths.append(os.path.join(path, filename))
            if "low" in filename:
                low_paths.append(os.path.join(path, filename))

        # remember to sort the lists to have correspondings paired together
        high_paths = sorted(high_paths)
        low_paths = sorted(low_paths)

        self.data = []
        for low, high in zip(low_paths, high_paths):  # just a check
            if low.split("low")[0] == high.split("high")[0]:
                self.data.append((low, high))
        print("Dataset loaded successfully!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        low_paths, high_paths = self.data[index]

        low_im = cv2.imread(low_paths, cv2.IMREAD_ANYDEPTH) / (2**16 - 1)
        high_im = cv2.imread(high_paths, cv2.IMREAD_ANYDEPTH) / (2**16 - 1)
        h, w = low_im.shape

        # simple rectification on the images when images less than the size of the crop
        # shouldn't be much of a problem
        if h <= self.crop_size:
            # print(low_im.shape, high_im.shape)
            h = self.crop_size + 1
            low_im = cv2.resize(low_im, (w, h)) # why does this work like this ???
            high_im = cv2.resize(high_im, (w, h))
            # print('new h', h)
            # print(low_im.shape, high_im.shape)

        if w <= self.crop_size:
            # print(low_im.shape, high_im.shape)
            w = self.crop_size + 1
            low_im = cv2.resize(low_im, (w, h))
            high_im = cv2.resize(high_im, (w, h))
            # print('new w', w)
            # print(low_im.shape, high_im.shape)


        # find a random crop with some details and shapes
        std_dev = 0
        trial = hyperparameters.sample_trial
        while std_dev < 0.15 and trial > 0:
            trial -= 1  # to avoid inf loop
            x = np.random.randint(0, w - self.crop_size)
            y = np.random.randint(0, h - self.crop_size)
            # print(x)
            low_crop = low_im[y : y + self.crop_size, x : x + self.crop_size]
            high_crop = high_im[y : y + self.crop_size, x : x + self.crop_size]
            std_dev = max(low_crop.std(), high_crop.std())
        # print(low_crop.shape, high_crop.shape)
        # print(low_im.shape, high_im.shape)
        low_crop = np.expand_dims(low_crop, 0)
        high_crop = np.expand_dims(high_crop, 0)
        return low_crop.astype(np.float16), high_crop.astype(np.float16)
    

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]