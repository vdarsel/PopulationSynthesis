import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps



class Beta_VAE(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, dim_in_cat, dim_in_cont, beta, in_z_dim=16):
        super(Beta_VAE, self).__init__()
        self.dim_in_cat = dim_in_cat
        self.dim_in = np.sum(dim_in_cat)+dim_in_cont
        self.dim_in_cat_sum = np.sum(self.dim_in_cat)
        self.dim_2 = len(dim_in_cat)+dim_in_cont
        self.dims = [max(2,int((len(dim_in_cat)+dim_in_cont)/np.power(2,i))) for i in range(3)]
        in_z_dim = max(in_z_dim,8)
        z_dim = 2*in_z_dim//8
        self.dims = [z_dim*2**(3-i) for i in range(4)]
        self.z_dim = self.dims[-1]
        print("Dimension of the layers (encoder): ", np.concatenate([[self.dim_in],self.dims]))
        print("Dimension of the layers (decoder): ", np.concatenate([self.dims[::-1],[self.dim_in]]))
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(self.dim_in,self.dims[0]),
            nn.ReLU(True),
            nn.Linear(self.dims[0], self.dims[1]),
            nn.ReLU(True),
            nn.Linear(self.dims[1], self.dims[2]),
            nn.ReLU(True),
            nn.Linear(self.dims[2], self.dims[3]*2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.dims[3], self.dims[2]),
            nn.ReLU(True),
            nn.Linear(self.dims[2], self.dims[1]),              
            nn.ReLU(True),
            nn.Linear(self.dims[1],self.dims[0] ),
            nn.ReLU(True),
            nn.Linear(self.dims[0], self.dim_in),  
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
    def get_embedding(self,x):
        return self._encode(x)
    
    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)
        
        x_recon_cat_tab = []
        j=0
        for dim_cat in self.dim_in_cat:
           x_recon_cat_tab.append(torch.softmax(x_recon[:,j:j+dim_cat],1))
           j += dim_cat
        x_recon_cat = torch.concat(x_recon_cat_tab,1)
        x_recon_cont = x_recon[:,self.dim_in_cat_sum:]

        return x_recon_cat, x_recon_cont, mu, logvar

    def loss(self, recon_x_cat, recon_x_cont, x_cat, x_cont, n_cat, mu, logvar):
        eps = 1e-15
        mse_loss = F.mse_loss(recon_x_cont, x_cont, reduction='mean')
        if ((x_cont).shape[1]==0):
            mse_loss=0
        ce_loss = (-torch.log(recon_x_cat+eps)*x_cat).sum(-1).mean()

        kl_diverge = -0.5*(1 + logvar - mu.pow(2) - logvar.exp()).mean()

        return (mse_loss + ce_loss + self.beta * kl_diverge)  # divide total loss by batch size
     
    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    
    def sample(self, n, device):
        z_val = torch.randn((n, self.z_dim)).to(device)
        z= self._decode(z_val)
        x_hat_cat_tab = []
        j=0
        for dim_cat in self.dim_in_cat:
           x_hat_cat_tab.append(torch.softmax(z[:,j:j+dim_cat],1))
           j += dim_cat
        x_hat_cont = z[:,self.dim_in_cat_sum:]

        return x_hat_cat_tab, x_hat_cont

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass