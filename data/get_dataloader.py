import torch
from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size, type_dataset):
    if type_dataset == "train":
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    elif type_dataset == "val":
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4)