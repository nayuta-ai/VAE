from torchvision import datasets, transforms


def transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    )


def get_dataset(transform, val_size):
    dataset_train = datasets.MNIST(
        "~/data/MNIST", train=True, download=True, transform=transform
    )
    dataset_val = datasets.MNIST(
        "~/data/MNIST", train=False, download=True, transform=transform
    )

    return dataset_train, dataset_val
