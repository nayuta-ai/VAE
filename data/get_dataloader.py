from torch import utils


def get_dataloader(dataset, batch_size, type_dataset):
    if type_dataset == "val":
        return utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    else:
        return utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4)