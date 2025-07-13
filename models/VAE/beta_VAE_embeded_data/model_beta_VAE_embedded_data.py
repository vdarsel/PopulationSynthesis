import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps



class Beta_VAE_Embedded_data(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""
    def __init__(self, dim_in, beta, in_z_dim=32):
        super(Beta_VAE_Embedded_data, self).__init__()
        self.z_dim = in_z_dim
        self.dim_in = dim_in
        self.beta = beta
        in_z_dim = max(in_z_dim,8)
        self.z_dim = 2*in_z_dim//8
        self.encoder = nn.Sequential(
            nn.Linear(dim_in,in_z_dim),
            # nn.ReLU(True),
            # nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(in_z_dim, in_z_dim//2),
            nn.ReLU(True),
            nn.Linear(in_z_dim//2, in_z_dim//4),
            nn.ReLU(True),
            nn.Linear(in_z_dim//4, self.z_dim*2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, in_z_dim//4),              
            nn.ReLU(True),
            nn.Linear(in_z_dim//4, in_z_dim//2),              
            nn.ReLU(True),
            nn.Linear(in_z_dim//2, in_z_dim),              
            nn.ReLU(True),
            nn.Linear(in_z_dim,dim_in),
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size
     
    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    
    def sample(self, n, device):
        z_val = torch.randn((n, self.z_dim)).to(device)
        return self._decode(z_val)

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