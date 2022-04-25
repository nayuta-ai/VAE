from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])


def get_dataset(transform, val_size):
    dataset = datasets.MNIST(
        '~/data/mnist', train=True, download=True, transform=transform)
    dataset_train, dataset_val = train_test_split(dataset, test_size=val_size)
    dataset_test = datasets.MNIST(
        '~/data/mnist', train=False, download=True, transform=transform)
    
    return dataset_train, dataset_val, dataset_test