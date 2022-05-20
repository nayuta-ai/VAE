from data.MNIST.get_dataloader import mnist_dataloader
from data.rubber.get_dataloader import rubber_dataloader


def get_dataloader(name, batch_size, type_dataset):
    if name == "mnist":
        return mnist_dataloader(batch_size=batch_size, type_dataset=type_dataset)
    
    elif name == "rubber":
        return rubber_dataloader(batch_size=batch_size, type_dataset=type_dataset)
