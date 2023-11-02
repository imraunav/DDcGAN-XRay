import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np


class XRayDataset(Dataset):
    def __init__(self, path, crop_size=84):
        super().__init__()
        self.crop_size = crop_size
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
        trial = 10
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
        return low_crop, high_crop
