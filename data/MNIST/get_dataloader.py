from torch import utils
from data.MNIST.get_dataset import transform, get_dataset


def mnist_dataloader(batch_size, type_dataset):
    trans = transform()
    dataset = get_dataset(transform=trans, type_dataset=type_dataset)
    if type_dataset == "train":
        return utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

    elif type_dataset == "val":
        return utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
