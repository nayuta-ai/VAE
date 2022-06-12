from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vae import Encoder, Decoder


class MultiVAE(nn.Module):
    def __init__(
        self,
        label_list: List[int],
        device="cpu",
        input_channel: int = 784,
        hidden_dims: List[int] = None,
        latent_dim: int = 10,
        distribution: str = "Bern",
    ):
        super().__init__()
        self.distribution = distribution
        self.label_list = label_list
        self.device = device
        self.encoder = Encoder(
            device=device,
            input_channel=input_channel,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            device=device,
            output_channel=input_channel,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )

        num_class = len(self.label_list)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, sig = self.encoder(x)
        z = self.reparametrize(mu, sig)
        y = self.decoder(z)
        t = self.classifier(z)
        return y, t, z

    def reparametrize(self, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        z = mu + torch.sqrt(sig) * torch.randn(mu.shape).to(self.device)
        return z

    def loss(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        mu, sig = self.encoder(x)
        z = self.reparametrize(mu, sig)
        y = self.decoder(z)
        t = self.classifier(z)
        L1 = -torch.mean(
            torch.sum(
                x * torch.log(y + 1e-08) + (1 - x) * torch.log(1 - y + 1e-08), dim=1
            )
        )
        L2 = -0.5 * torch.mean(torch.sum(1 + torch.log(sig) - mu**2 - sig, dim=1))
        criterion = nn.CrossEntropyLoss()
        L3 = criterion(t, label)
        L = L1 + L2 + L3
        return L
    
    def accuracy(self, x: torch.Tensor, label: torch.Tensor):
        _, t, _ = self.forward(x)
        acc = 0
        data_length = len(t)
        for i in range(data_length):
            if self.label_list[torch.argmax(t[i]).item()] == label[i]:
                acc += 1
        return acc / data_length * 100