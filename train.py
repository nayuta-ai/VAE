import yaml
from comet_ml import Experiment
from torchvision import datasets, transforms
import torch
from torch import optim, utils
import setting
from model.vae import VAE
from trainer import train, test
from data.MNIST.get_dataset import transform, get_dataset
from data.get_dataloader import get_dataloader
from visualization.generate import generate, random_generate


def main():
    with open("config/vae.yaml", "r") as yml:
        config = yaml.load(yml)["Model"]
    # Create an experiment with your api key
    experiment = Experiment(
        api_key=setting.API_KEY,
        project_name=config["name"],
        workspace=setting.WORKSPACE,
    )
    experiment.log_parameters(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    trans = transform()
    dataset_train, dataset_val = get_dataset(transform=trans, val_size=0.1)

    dataloader_train = get_dataloader(
        dataset_train, batch_size=config["batch_size"], type_dataset="train")
    dataloader_val = get_dataloader(
        dataset_val, batch_size=config["batch_size"], type_dataset="val")
    
    # model
    model = VAE(
        device=device,
        input_channel=config["input_channel"],
        latent_dim=config["latent_dim"],
        distribution=config["distribution"]
    ).to(device)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    train(
        model=model, dataloader_train=dataloader_train, dataloader_val=dataloader_val, 
        optimizer=optimizer, device=device, iteration=config["num_epochs"], 
        experiment=experiment
    )
    
    test(
        model=model, dataloader=dataloader_val, device=device, experiment=experiment
    )
    
    if config["latent_dim"] == 2:
        generate(
            model=model, distribution=config["distribution"], device=device, experiment=experiment
        )
    random_generate(
        model=model, distribution=config["distribution"], latent_dim=config["latent_dim"], device=device, experiment=experiment
    )


if __name__ == "__main__":
    main()