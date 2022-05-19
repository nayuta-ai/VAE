import os
from typing import List

import albumentations as album
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(
            self, dataset: List[str], csv_file: pd.DataFrame,
            transform: album.Compose) -> None:
        """ Initialization
        Args:
            dataset (List[str]): original dataset
            csv_file (pd.DataFrame): file contained the detail of the dataset
            transform (album.Compose): the procedure of the data augmentation
        """
        self.dataset = dataset
        self.csv = csv_file
        self.transform = transform
        self.as_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        """ Function to count the number of data
        Returns:
            int: the number of files
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> (str, torch.Tensor, np.float32):
        """ Function to get item
        Args:
            idx(int): the index of files
        Returns:
            str: the name of item
            torch.Tensor: the image data formatted Tensor
            np.float32: the median of data
        """
        img_data = self.dataset[idx]
        data = self.csv
        img = cv2.imread(img_data, 0)
        img_data_path = os.path.basename(img_data)
        perf = np.nanmedian(
            data[data['name'] == img_data_path[:10]]['Resist']
            ).astype(np.float32)

        # There are two types: "STEM," which is just an ordinary SEM image,
        # and "STEM-EDX," which contains information on the distribution of
        # each of the nine elements.
        # "STEM-EDX" data has a "T" in the file name.
        if 'T' in img_data_path:
            img = ((img - 138.8) * 0.6 + 142.7).astype('uint8')

        augments = self.transform(image=img)
        img = self.as_tensor(augments['image'])
        return img_data_path, img, perf
