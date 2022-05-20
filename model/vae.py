from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        device="cpu",
        input_channel: int = 784,
        hidden_dims: List[int] = None,
        latent_dim: int = 10,
        distribution: str = "Gauss",
    ):
        super().__init__()
        self.distribution = distribution
        self.device = device
        self.encoder = Encoder(
            device=device,
            input_channel=input_channel,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
        if self.distribution == "Bern":
            self.decoder = Decoder(
                device=device,
                output_channel=input_channel,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
            )
        else:
            self.decoder = GaussDecoder(
                device=device,
                output_channel=input_channel,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, sig = self.encoder(x)
        z = self.reparametrize(mu, sig)
        if self.distribution == "Bern":
            y = self.decoder(z)
        else:
            mu, sig = self.decoder(z)
            y = self.reparametrize(mu, sig)
        return y, z

    def reparametrize(self, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        z = mu + torch.sqrt(sig) * torch.randn(mu.shape).to(self.device)
        return z

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        mu, sig = self.encoder(x)
        z = self.reparametrize(mu, sig)
        if self.distribution == "Bern":
            y = self.decoder(z)
            L1 = -torch.mean(
                torch.sum(
                    x * torch.log(y + 1e-08) + (1 - x) * torch.log(1 - y + 1e-08), dim=1
                )
            )
        else:
            dec_mu, dec_sig = self.decoder(z)
            y = self.reparametrize(dec_mu, dec_sig)
            criterion = nn.MSELoss()
            L1 = criterion(y, x) * 0.5
        L2 = -0.5 * torch.mean(torch.sum(1 + torch.log(sig) - mu**2 - sig, dim=1))
        L = L1 + L2
        return L


class Encoder(nn.Module):
    def __init__(
        self,
        device,
        input_channel: int = 784,
        hidden_dims: List[int] = None,
        latent_dim: int = 10,
    ) -> None:
        super().__init__()
        self.device = device
        if not hidden_dims:
            hidden_dims = [512, 256, 128, 64, 32]
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(nn.Linear(input_channel, h_dim), nn.LeakyReLU())
            )
            input_channel = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mean = nn.Linear(h_dim, latent_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(h_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.encoder(x)
        mu = self.fc_mean(res)
        sig = F.softplus(self.fc_var(res))
        return mu, sig


class Decoder(nn.Module):
    def __init__(
        self,
        device,
        output_channel: int = 784,
        hidden_dims: List[int] = None,
        latent_dim: int = 10,
    ) -> None:
        super().__init__()
        self.device = device
        if not hidden_dims:
            hidden_dims = [512, 256, 128, 64, 32]
        hidden_dims.reverse()

        modules = []
        input_dim = latent_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(input_dim, h_dim), nn.LeakyReLU()))
            input_dim = h_dim
        self.decoder = nn.Sequential(*modules)
        self.fc_last = nn.Sequential(nn.Linear(input_dim, output_channel), nn.Sigmoid())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        res = self.decoder(z)
        y = self.fc_last(res)
        return y


class GaussDecoder(nn.Module):
    def __init__(
        self,
        device,
        output_channel: int = 784,
        hidden_dims: List[int] = None,
        latent_dim: int = 10,
    ) -> None:
        super().__init__()
        self.device = device
        if not hidden_dims:
            hidden_dims = [512, 256, 128, 64, 32]
        hidden_dims.reverse()

        modules = []
        input_dim = latent_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(input_dim, h_dim), nn.LeakyReLU()))
            input_dim = h_dim
        self.decoder = nn.Sequential(*modules)
        self.fc_mean = nn.Linear(input_dim, output_channel)
        self.fc_var = nn.Sequential(
            nn.Linear(input_dim, output_channel),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        mu = self.fc_mean(h)
        sig = F.softplus(self.fc_var(h))
        return mu, sig
