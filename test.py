import yaml
import torch
from model.vae import VAE
from data import get_dataloader
from visualization import visualize_z_label


def main():
    with open("config/vae.yaml", "r") as yml:
        config = yaml.load(yml)["Model"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = VAE(
        device=device,
        input_channel=config["input_channel"],
        latent_dim=config["latent_dim"],
        distribution=config["distribution"],
    ).to(device)
    
    model.load_state_dict(torch.load("result/mnist_1935.pth"))
    dataloader_val = get_dataloader(
        name=config["dataset"], batch_size=config["batch_size"], type_dataset="val"
    )
    visualize_z_label(model, dataloader_val, device)

if __name__ == "__main__":
    main()