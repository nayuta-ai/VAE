import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, device, beta: int = 1, in_channels: int = 1, z_dim: int = 10):
      super(VAE, self).__init__()
      self.device = device
      self.beta = beta
      # encoder
      modules = []
      hidden_dims = [32, 64, 128, 256, 512]
      for h_dim in hidden_dims:
            modules.append(
              nn.Sequential(
                nn.Conv2d(
                  in_channels, out_channels=h_dim,
                  kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
              )
            )
            in_channels = h_dim
      self.encoder = nn.Sequential(*modules)
      self.dense_encmean = nn.Linear(hidden_dims[-1]*49, z_dim)
      self.dense_encvar = nn.Linear(hidden_dims[-1]*49, z_dim)
      # decoder
      modules = []
      self.decoder_input = nn.Linear(z_dim, hidden_dims[-1]*49)
      hidden_dims.reverse()
      for i in range(len(hidden_dims) - 1):
            modules.append(
              nn.Sequential(
                nn.ConvTranspose2d(
                  hidden_dims[i], hidden_dims[i+1], kernel_size=3,
                  stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU(),
              )
            )
      self.decoder = nn.Sequential(*modules)

      self.final_layer = nn.Sequential(
        nn.ConvTranspose2d(
          hidden_dims[-1], hidden_dims[-1], kernel_size=3,
          stride=2, padding=1, output_padding=1
        ),
        nn.BatchNorm2d(hidden_dims[-1]),
        nn.LeakyReLU(),
        nn.Conv2d(
          hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
        nn.Tanh(),
      )
    
    def _encoder(self, x):
      result = self.encoder(x)
      result = torch.flatten(result, start_dim=1)
      mu = self.dense_encmean(result)
      var = F.softplus(self.dense_encvar(result))
      return mu, var
    
    def _sample_z(self, mu, log_var):
      epsilon = torch.randn(mu.shape).to(self.device)
      std = torch.exp(0.5 * log_var)
      return mu + std*epsilon
 
    def _decoder(self, z):
      result = self.decoder_input(z)
      result = result.view(-1, 512, 7, 7)
      result = self.decoder(result)
      result = self.final_layer(result)
      return result

    def forward(self, x):
      mu, log_var = self._encoder(x)
      z = self._sample_z(mu, log_var)
      x = self._decoder(z)
      return x, z
    
    def loss(self, x):
      mu, log_var = self._encoder(x)
      KL = -0.5 * torch.mean(torch.sum(1 + log_var - mu**2 - torch.exp(log_var)))
      z = self._sample_z(mu, log_var)
      y = self._decoder(z)
      recons_loss = F.mse_loss(y, x)
      lower_bound = [-KL, recons_loss]
      return -sum(lower_bound)
