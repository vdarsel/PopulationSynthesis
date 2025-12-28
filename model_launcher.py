import os

from models.Embedding.Transformer_VAE.train_Transformer_VAE import learn_encoding_Transformer_VAE
from models.Embedding.Transformer_VAE.sample_Transformer_VAE import decode_Transformer_VAE

from models.Diffusion.Diffusion_Embedded_data.train_Diffusion_Embedded_Data import train_Diffusion_from_embedded_data
from models.Diffusion.Diffusion_Embedded_data.sample_Diffusion_Embedded_Data import sample_Diffusion_from_embedded_data

from models.VAE.TVAE.train_TVAE import train_TVAE
from models.VAE.TVAE.sample_TVAE import sample_TVAE


from models.GAN.WGAN.train_WGAN_alone import train_WGAN_alone
from models.GAN.WGAN.sample_WGAN_alone import sample_WGAN_alone

from models.GAN.WGAN_embeded_data.train_WGAN_embedded import train_WGAN_embedding_data
from models.GAN.WGAN_embeded_data.sample_WGAN_embedded import sample_WGAN_embedding_data

from models.VAE.beta_VAE.train_beta_VAE_alone import train_beta_VAE_alone
from models.VAE.beta_VAE.sample_beta_VAE_alone import sample_beta_VAE_alone

from models.VAE.beta_VAE_embeded_data.train_beta_VAE_embedded_data import train_beta_VAE_from_embedded_data
from models.VAE.beta_VAE_embeded_data.sample_beta_VAE_embedded_data import sample_beta_VAE_from_embedded_data

from models.Bayesian_Network.Hill_Climbing.BN_Hill_climb import train_sample_BN_hill_climb

from models.Bayesian_Network.Tree.BN_tree import train_sample_BN_tree

from models.Monte_Carlo_Markov_Chain.Frequentist.MCMC_frequentist import MCMC_frequentist_learn_sample

from models.Monte_Carlo_Markov_Chain.Bayesian.MCMC_Bayesian import MCMC_Bayesian_learn_sample

from models.Direct_Inflating.Direct_Inflating import train_sample_Direct_Inflating

def learn_transformer_vae_if_needed(config):
    if (config.force_training_embedding) or (not os.path.isfile(f"ckpt/{config.folder_save}/train_z.npy")):
        learn_encoding_Transformer_VAE(config)


class Model_for_Population_Synthesis:
    def __init__(self):
        pass
    
    def train_if_needed(self, config):
        pass
    
    def sample(self, config):
        pass
    
    def termination_saving(self):    
        pass


#####################################################
###                   Diffusion                   ###
#####################################################

class Diffusion_model(Model_for_Population_Synthesis):
    
    def __init__(self, projection_dim):
        self.dim = projection_dim
        
    def train_if_needed(self, config):
        learn_transformer_vae_if_needed(config)
        
        if (config.force_training) or (not os.path.isfile(f"ckpt/{config.folder_save}/model{self.termination_saving()}.pt")):
            train_Diffusion_from_embedded_data(config, self.dim)
    
    def sample(self, config):
        sample_Diffusion_from_embedded_data(config, self.dim)
        
        decode_Transformer_VAE(config,self.termination_saving())

    def termination_saving(self):    
        return f"_diffusion_{self.dim}"
    

#####################################################
###                     VAE                       ###
#####################################################


class beta_VAE_embedding(Model_for_Population_Synthesis):
    def __init__(self, beta, dim):
        self.beta = beta
        self.dim = dim

    def train_if_needed(self, config):
        learn_transformer_vae_if_needed(config) 
    
        if (config.force_training) or (not os.path.isfile(f"ckpt/{config.folder_save}/model{self.termination_saving()}.pt")):
            train_beta_VAE_from_embedded_data(config, self.beta, self.termination_saving(), self.dim)
        
    def sample(self, config):
        sample_beta_VAE_from_embedded_data(config, self.beta, self.termination(), self.dim)
        
        decode_Transformer_VAE(config,self.termination_saving())

    def termination_saving(self):    
        return f"_beta_VAE_(embedding)_beta_{f"{self.beta}".replace(".","_")}_{self.dim}"
        
    
class beta_VAE(Model_for_Population_Synthesis):
    def __init__(self, beta, dim):
        self.beta = beta
        self.dim = dim
    
    def train_if_needed(self, config):
        if (config.force_training) or (not os.path.isfile(f"ckpt/{config.folder_save}/model{self.termination_saving()}.pt")):
            train_beta_VAE_alone(config, self.beta, self.termination_saving(),  self.dim)
        
    def sample(self, config):
        sample_beta_VAE_alone(config, self.beta, self.termination_saving(), self.dim)
        
    def termination_saving(self):
        return f"_beta_VAE_beta_{f"{self.beta}".replace(".","_")}_{self.dim}"

class TVAE(Model_for_Population_Synthesis):
    def __init__(self, beta, dim):
        self.beta = beta
        self.dim = dim

    def train_if_needed(self, config):
        if (config.force_training) or (not os.path.isfile(f"ckpt/{config.folder_save}/encoder_TVAE_dim_{self.dim}_beta_{f"{self.beta}".replace(".","_")}.pt")):
            train_TVAE(config, self.beta, self.dim)
    
    def sample(self, config):
        sample_TVAE(config, self.beta, self.termination_saving(), self.dim)
    
    def termination_saving(self):
        return f"_TVAE_beta_{f"{self.beta}".replace(".","_")}_{self.dim}"

#####################################################
###                     GAN                       ###
#####################################################

class WGAN_embedding(Model_for_Population_Synthesis):
    def __init__(self, dim):
        self.dim = dim
    
    def train_if_needed(self, config):
        learn_transformer_vae_if_needed(config)
        if (config.force_training) or (not os.path.isfile(f"ckpt/{config.folder_save}/discriminator{self.termination_saving()}.pt")):
            train_WGAN_embedding_data(config, self.termination_saving(), self.dim)
            
    def sample(self, config):
        sample_WGAN_embedding_data(config, self.termination_saving(), self.dim)
        decode_Transformer_VAE(config,self.termination_saving())
    
    def termination_saving(self):
        return f"_WGAN_(embedding)_{self.dim}"

class WGAN(Model_for_Population_Synthesis):
    def __init__(self, dim):
        self.dim = dim
        
    def train_if_needed(self, config):
        if (config.force_training) or (not os.path.isfile(f"ckpt/{config.folder_save}/discriminator{self.termination_saving()}.pt")):
            train_WGAN_alone(config, self.termination_saving(), self.dim)
    
    def sample(self, config):
        sample_WGAN_alone(config, self.termination_saving(), self.dim)

    def termination_saving(self):
        return f"_WGAN_{self.dim}"

#####################################################
###                    MCMC                       ###
#####################################################

class MCMC_frequentist(Model_for_Population_Synthesis):
    def sample(self, config):
        MCMC_frequentist_learn_sample(config)
    def termination_saving(self):
        return "_MCMC"


class MCMC_Bayesian(Model_for_Population_Synthesis):
    def __init__(self, alpha):
        self.alpha = alpha
    def sample(self, config):
        MCMC_Bayesian_learn_sample(config, self.alpha, self.termination_saving())
    def termination_saving(self):
        return f"_MCMC_Bayesian_{f"{self.alpha}".replace(".","_")}"

#####################################################
###              Bayesian Network                 ###
#####################################################

class Bayesian_Network_hill(Model_for_Population_Synthesis):
    def sample(self, config):
        train_sample_BN_hill_climb(config)
    def termination_saving(self):
        return "_Bayesian_Network_Hill_Climb"


class Bayesian_Network_tree(Model_for_Population_Synthesis):
    def sample(self, config):
        train_sample_BN_tree(config)
    def termination_saving(self):
        return "_Bayesian_Network_Tree"
    
    
#####################################################
###              Direct Inflating                 ###
#####################################################

class Direct_Inflating(Model_for_Population_Synthesis):
    def sample(self, config):
        train_sample_Direct_Inflating(config)
    def termination_saving(self):
        return "_Direct_Inflating"
    