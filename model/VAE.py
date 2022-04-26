import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, device, vertical=28, side=28, enc1=200, enc2=200, z_dim=10, dec1=200, dec2=200):
      super(VAE, self).__init__()
      self.dense_enc1 = nn.Linear(vertical*side, enc1)
      self.dense_enc2 = nn.Linear(enc1, enc2)
      self.dense_encmean = nn.Linear(enc2, z_dim)
      self.dense_encvar = nn.Linear(enc2, z_dim)
      self.dense_dec1 = nn.Linear(z_dim, dec1)
      self.dense_dec2 = nn.Linear(dec1, dec2)
      self.dense_dec3 = nn.Linear(dec2, vertical*side)
      self.device = device
    
    def _encoder(self, x):
      x = F.relu(self.dense_enc1(x))
      x = F.relu(self.dense_enc2(x))
      mean = self.dense_encmean(x)
      var = F.softplus(self.dense_encvar(x))
      return mean, var
    
    def _sample_z(self, mean, var):
      epsilon = torch.randn(mean.shape).to(self.device)
      return mean + torch.sqrt(var) * epsilon
 
    def _decoder(self, z):
      x = F.relu(self.dense_dec1(z))
      x = F.relu(self.dense_dec2(x))
      x = F.sigmoid(self.dense_dec3(x))
      return x

    def forward(self, x):
      mean, var = self._encoder(x)
      z = self._sample_z(mean, var)
      x = self._decoder(z)
      return x, z
    
    def loss(self, x):
      mean, var = self._encoder(x)
      KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var))
      z = self._sample_z(mean, var)
      y = self._decoder(z)
      reconstruction = torch.mean(torch.sum(x * torch.log(y + 1e-8) + (1 - x) * torch.log(1 - y + 1e-8)))
      lower_bound = [-KL, reconstruction]                                      
      return -sum(lower_bound)
