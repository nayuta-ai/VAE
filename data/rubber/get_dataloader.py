from typing import List
import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config.path import DATA_PATH
from data.rubber.get_dataset import ImageDataset


def rubber_dataloader(
        batch_size: int, type_dataset: str) -> DataLoader:
    """ A function to load data
    Args:
        batch_size (int): the size of batch
        type_dataset (str): the type of dataset such as train, val, and test
    """
    # read data
    dataset = glob.glob(DATA_PATH)
    data = ImageDataset(dataset=dataset)

    if type_dataset == "train":
        return torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True, num_workers=4)
    elif type_dataset == "val":
        return torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False, num_workers=4)