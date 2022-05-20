from torchvision import datasets, transforms


def transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    )


def get_dataset(transform, type_dataset: str):
    if type_dataset == "train":
        return datasets.MNIST(
            "~/data/MNIST", train=True, download=True, transform=transform
        )
    elif type_dataset == "val":
        return datasets.MNIST(
            "~/data/MNIST", train=False, download=True, transform=transform
        )
