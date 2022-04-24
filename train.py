import hydra
from omegaconf import DictConfig
from comet_ml import Experiment
from torchvision import datasets, transforms
import torch
from torch import optim, utils
from model.VAE import VAE
from trainer.trainer import train, val


def main():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="KJaIbULChXRLi1jMjxzip9Cog",
        project_name="vae",
        workspace="nayuta-ai",
    )
    hyper_params = {
        "hidden_dim": 10,
        "batch_size": 1000,
        "num_epochs": 20,
        "learning_rate": 0.01
    }
    experiment.log_parameters(hyper_params)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x.view(-1))])

    dataset_train = datasets.MNIST(
        '~/data/mnist', 
        train=True, 
        download=True, 
        transform=transform)
    dataset_valid = datasets.MNIST(
        '~/data/mnist', 
        train=False, 
        download=True, 
        transform=transform)

    dataloader_train = utils.data.DataLoader(dataset_train,
                                            batch_size=hyper_params["batch_size"],
                                            shuffle=True,
                                            num_workers=4)
    dataloader_valid = utils.data.DataLoader(dataset_valid,
                                            batch_size=hyper_params["batch_size"],
                                            shuffle=True,
                                            num_workers=4)
    
    # model
    model = VAE(z_dim=hyper_params["hidden_dim"], device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])
    
    train(
        model=model, dataloader=dataloader_train, optimizer=optimizer, 
        device=device, iteration=hyper_params["num_epochs"], experiment=experiment)
    
    val(model=model, dataloader=dataloader_valid, device=device, experiment=experiment)


if __name__ == "__main__":
    main()