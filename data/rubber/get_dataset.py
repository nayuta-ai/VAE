import os
import glob
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from config.path import DATA_PATH


class ImageDataset(Dataset):
    def __init__(self) -> None:
        """ Initialization
        Args:
            dataset (Tuple): original dataset
        """
        dataset = glob.glob(DATA_PATH)
        self.dataset = dataset

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
        img, label = torch.load(img_data)
        img = img.reshape(28*28)
        return img, label
