from comet_ml import Experiment
import torch
from torch import optim, utils
from model.VAE import VAE
from trainer.trainer import train, test
from data.rubber.get_dataloader import get_dataloader


def main():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="KJaIbULChXRLi1jMjxzip9Cog",
        project_name="vae",
        workspace="nayuta-ai",
    )
    hyper_params = {
        "input_vertical_size": 224,
        "input_side_size": 224,
        "hidden_dim": 10,
        "batch_size": 16,
        "num_epochs": 1,
        "learning_rate": 0.001
    }
    experiment.log_parameters(hyper_params)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    dataloader_train = get_dataloader(
        batch_size=hyper_params["batch_size"], type_dataset="train")
    dataloader_val = get_dataloader(
        batch_size=hyper_params["batch_size"], type_dataset="val")
    
    # model
    model = VAE(
        z_dim=hyper_params["hidden_dim"], device=device).to(device)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])
    
    train(
        model=model, dataloader_train=dataloader_train, dataloader_val=dataloader_val, 
        optimizer=optimizer, device=device, iteration=hyper_params["num_epochs"], 
        experiment=experiment)
    
    test(
        model=model, vertical=hyper_params["input_vertical_size"], side=hyper_params["input_side_size"], 
        dataloader=dataloader_val, device=device, experiment=experiment)


if __name__ == "__main__":
    main()