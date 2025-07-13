import torch
import torch.nn as nn
from torch import autograd

class Generator(nn.Module):
    def __init__(self, dim_data_num, k=1):
        super(Generator, self).__init__()

        self.dim = dim_data_num
        self.k = k

        self.features_to_embedding = nn.Sequential(
            nn.Linear(self.k * 4, self.k * 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.k * 2),
            nn.Linear(self.k * 2,self.k),
            nn.ReLU(),
            nn.BatchNorm1d(self.k),
            nn.Linear(self.k,self.dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim),
            nn.Linear(self.dim,self.dim)
        )
    
    def forward(self, input_data):
        # Get embedding
        return self.features_to_embedding(input_data)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples,self.k * 4))


class Discriminator(nn.Module):
    def __init__(self, dim_num, k=1):

        super(Discriminator, self).__init__()


        self.dim = dim_num
        self.k = k
        self.sample_to_features = nn.Sequential(
            nn.Linear(self.dim,self.k),
            nn.LeakyReLU(),
            nn.Linear(self.k, self.k * 2),
            nn.LeakyReLU(),
            nn.Linear(self.k * 2, self.k * 4),
        )

        self.features_to_prob = nn.Sequential(
            nn.Linear(self.k * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        x = self.sample_to_features(input_data)
        return self.features_to_prob(x)
    

def calc_gradient_penalty(netD, real_data, fake_data, device, lambda_=10):
    batch_size, dim = real_data.shape 
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, dim).contiguous()
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty
