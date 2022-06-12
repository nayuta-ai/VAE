from torchvision import datasets, transforms


def transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    )


def get_mnist(type_dataset: str):
    trans = transform()
    if type_dataset == "train":
        return datasets.MNIST(
            "~/data/MNIST", train=True, download=True, transform=trans
        )
    elif type_dataset == "val":
        return datasets.MNIST(
            "~/data/MNIST", train=False, download=True, transform=trans
        )
