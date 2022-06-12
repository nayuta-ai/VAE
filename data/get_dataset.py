import torch.utils.data

from data.MNIST.diff_mnist_dataset import MNISTLabel, create_multi_label_dataset
from data.MNIST.get_dataset import get_mnist
from data.rubber.get_dataset import ImageDataset


def get_dataset(dataset_name: str, label_dict = None):
    if dataset_name == "mnist":
        return get_mnist("train"), get_mnist("val")
    
    elif dataset_name == "diff_mnist":
        data_train, data_val = get_mnist("train"), get_mnist("val")
        data_train = create_multi_label_dataset(data_train, label_dict)
        data_val = create_multi_label_dataset(data_val, label_dict)
        data_train = MNISTLabel(data_train)
        data_val = MNISTLabel(data_val)
        return data_train, data_val
    
    elif dataset_name == "rubber":
        dataset = ImageDataset()
        n_sample = len(dataset)
        train_size = int(n_sample * 0.8)
        val_size = n_sample - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        return train_dataset, val_dataset