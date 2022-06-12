import yaml
from comet_ml import Experiment
import torch
from torch import optim

import setting
from model.vae import VAE
from model.classifier import Classifier
from data import get_dataset, get_dataloader
from trainer_classifier import train


def main():
    with open("config/vae.yaml", "r") as yml:
        config = yaml.load(yml)["Model"]
    
    experiment = Experiment(
        api_key=setting.API_KEY,
        project_name=config["name"],
        workspace=setting.WORKSPACE,
    )
    experiment.log_parameters(config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_list = config["label_list"]
    label_dict = {}
    label_list_mod = range(len(label_list))
    for i, label in enumerate(label_list):
        label_dict[label] = i
    data_train, data_val = get_dataset(dataset_name="diff_mnist", label_dict=label_dict)
    dataloader_train = get_dataloader(
        data_train, batch_size=config["batch_size"], type_dataset="train"
    )
    dataloader_val = get_dataloader(
        data_val, batch_size=config["batch_size"], type_dataset="val"
    )
    
    model = VAE(
        device=device,
        input_channel=config["input_channel"],
        latent_dim=config["latent_dim"],
        distribution=config["distribution"],
    ).to(device)

    discriminator = Classifier(
        device=device,
        input_channel=config["latent_dim"],
        label_list=label_list
    ).to(device)
    
    model.load_state_dict(torch.load("result/mnist_1935.pth"))
    optimizer = optim.Adam(discriminator.parameters(), lr=config["learning_rate"])
    train(
        generater=model,
        model=discriminator,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        optimizer=optimizer,
        device=device,
        iteration=config["num_epochs"],
        experiment=experiment,
    )

if __name__ == "__main__":
    main()