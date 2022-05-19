from typing import List
import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config.path import DATA_PATH, CSV_PATH
from data.rubber.get_dataset import ImageDataset
from data.rubber.transform import train_transform, val_transform


def get_dataloader(
        batch_size: int, type_dataset: str) -> DataLoader:
    """ A function to load data
    Args:
        dataset (List[str]): original dataset
        csv_file (pd.DataFrame): file contained the detail of original dataset
        batch_size (int): the size of batch
        type_dataset (str): the type of dataset such as train, val, and test
    """
    # read data
    dataset = glob.glob(DATA_PATH)
    csv_file = pd.read_csv(CSV_PATH)

    if type_dataset == "train":
        data = ImageDataset(
            dataset=dataset, csv_file=csv_file, transform=train_transform())
        return torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True, num_workers=4)
    elif type_dataset == "val":
        data = ImageDataset(
            dataset=dataset, csv_file=csv_file, transform=val_transform())
        return torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False, num_workers=4)
