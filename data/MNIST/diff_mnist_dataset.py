import torch

from data.MNIST.get_dataset import get_mnist


def create_multi_label_dataset(dataset, label_dict):
    multi_label_data = []
    for data, label in dataset:
        if label in label_dict:
            multi_label_data.append((data, label_dict[label]))
    return multi_label_data


class MNISTLabel:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]