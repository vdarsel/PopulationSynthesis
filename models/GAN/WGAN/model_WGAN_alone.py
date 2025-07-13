import torch
import torch.nn as nn
import numpy as np
from torch import autograd

class Generator_Tabular(nn.Module):
    def __init__(self, dim_data_cat_list, dim_data_num, k=32):
        super(Generator_Tabular, self).__init__()

        self.dim = len(dim_data_cat_list)+dim_data_num
        self.dim_last = np.sum(dim_data_cat_list)+dim_data_num
        self.dim_cat = dim_data_cat_list
        self.dim_cat_sum = np.sum(dim_data_cat_list)
        self.dim_num = dim_data_num
        self.k = k

        self.features_to_data = nn.Sequential(
            nn.Linear(self.k*4, self.k*2),
            nn.ReLU(),
            nn.BatchNorm1d(self.k * 2),
            nn.Linear(self.k * 2,self.k),
            nn.ReLU(),
            nn.BatchNorm1d(self.k),
            nn.Linear(self.k,self.dim_last),
        )
    
    def forward(self, input_data):
        # Get embedding
        data = self.features_to_data(input_data)
        x_cat_tab = []
        j=0
        for dim_cat in self.dim_cat:
           x_cat_tab.append(torch.softmax(data[:,j:j+dim_cat],1))
           j += dim_cat
        x_cat = torch.concat(x_cat_tab,1)
        x_num = data[:,self.dim_cat_sum:]
        return x_num, x_cat

    def sample_latent(self, num_samples):
        return torch.randn((num_samples,self.k * 4))
    
    def sample(self, n_samples, device):
        latent = self.sample_latent(n_samples).to(device)
        data = self.features_to_data(latent)
        x_cat_tab = []
        j=0
        for dim_cat in self.dim_cat:
           x_cat_tab.append(torch.softmax(data[:,j:j+dim_cat],1))
           j += dim_cat
        # x_cat = torch.concat(x_cat_tab,1)
        x_num = data[:,self.dim_cat_sum:]
        return x_cat_tab, x_num


class Discriminator_Tabular(nn.Module):
    def __init__(self, dim_data_cat_list, dim_data_num, k=32):

        super(Discriminator_Tabular, self).__init__()
        self.dim = dim_data_num+np.sum(dim_data_cat_list)
        self.k = k
        self.sample_to_features = nn.Sequential(
            nn.Linear(self.dim, k),
            nn.LeakyReLU(),
            nn.Linear(k,k*2),
            nn.LeakyReLU(),
            nn.Linear(k*2, k*4),
        )

        self.features_to_prob = nn.Sequential(
            nn.Linear(self.k*4, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x = self.sample_to_features(data)
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
