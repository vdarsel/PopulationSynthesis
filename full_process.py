import os
import argparse
import yaml

import torch

from conf_fctn import odict,dict2namespace

from model_launcher import *

from evaluation.evaluation_from_parameters import full_evaluation


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Population Synthesis')
    
    parser.add_argument('--variable', type=str, required=False, default=None)
    parser.add_argument('--size_data', type=float, required=False, default=1)
    
    parser.add_argument("--Diffusion", action='store_true')
    parser.add_argument("--TVAE", action='store_true')
    parser.add_argument("--WGAN", action='store_true')
    parser.add_argument("--WGAN_embedded", action='store_true')
    parser.add_argument("--beta_VAE", action='store_true')
    parser.add_argument("--beta_VAE_embedded", action='store_true')
    parser.add_argument("--BN_hill", action='store_true')
    parser.add_argument("--BN_tree", action='store_true')
    parser.add_argument("--MCMC_frequence", action='store_true')
    parser.add_argument("--MCMC_Bayesian", action='store_true')
    
    
    parser.add_argument("--no_evaluation", action='store_true')
    
    parser.add_argument("--Diffusion_dim", type=int, default=2000)
    parser.add_argument("--TVAE_dim", type=int, default=2000)
    parser.add_argument("--TVAE_beta", type=int, default=1)
    parser.add_argument("--beta_VAE_dim", type=int, default=2000)
    parser.add_argument("--beta_VAE_beta", type=int, default=1)
    parser.add_argument("--beta_VAE_embedded_dim", type=int, default=2000)
    parser.add_argument("--beta_VAE_embedded_beta", type=int, default=1)
    parser.add_argument("--WGAN_dim", type=int, default=2000)
    parser.add_argument("--WGAN_embedded_dim", type=int, default=2000)
    

    args = odict(vars(parser.parse_args()))
    # print(args)
    
    with open(f"conf/conf_variable/{args.variable}.yml", "r") as f:
        config_ = yaml.safe_load(f)
    config = dict2namespace(config_)

    if(args.size_data == int(args.size_data)):
        str_float = "_".join(str(args.size_data).split(".0")[0].split("."))
    else:
        str_float = "_".join(str(args.size_data).split("."))
    
    with open(f"conf/conf_size/{str_float}%.yml", "r") as f:
        config_ = yaml.safe_load(f)
    config_2 = dict2namespace(config_)

    for key,val in config_2._get_kwargs():
        setattr(config, key, val)
        
    setattr(config,"folder_save",config.folder_save_start+config.folder_save_end)
    # setattr(config,"filename",config.filename_start.split(".csv")[0]+config.filename_end+".csv")
    setattr(config,"variable",args.variable)
    setattr(config,"str_float", str_float)

    # check cuda
    if config.gpu != -1 and torch.cuda.is_available():
        config.device = 'cuda:{}'.format(config.gpu)
    else:
        config.device = 'cpu'
    print("device:", config.device)
    
    models = []
    
    if (args.Diffusion):
        model = Diffusion_model(args.Diffusion_dim)
        models.append(model)

    if (args.TVAE):
        model = TVAE(args.TVAE_beta, args.TVAE_dim)
        models.append(model)

    if (args.beta_VAE):
        model = beta_VAE(args.beta_VAE_beta, args.beta_VAE_dim)
        models.append(model)

    if (args.beta_VAE_embedded):
        model = beta_VAE_embedding(args.beta_VAE_embedded_beta, args.beta_VAE_embedded_dim)
        models.append(model)
        
    if (args.WGAN):
        model = WGAN(args.WGAN_dim)
        models.append(model)

    if (args.WGAN_embedded):
        model = WGAN(args.WGAN_embedded_dim)
        models.append(model)
        
    if(args.BN_hill):
        model = Bayesian_Network_hill()
        models.append(model)
        
    if(args.BN_tree):
        model = Bayesian_Network_tree()
        models.append(model)
        
    if(args.MCMC_frequence):
        model = MCMC_frequentist()
        models.append(model)
        
    if(args.MCMC_Bayesian):
        model = MCMC_Bayesian()
        models.append(model)
        
        
    print("\n\n\n***************List of models***************")
    for model in models:
        print(model.terminaison_saving())
    print("********************************************\n\n\n")

    for model in models:
        print(f"\n\n***Current Model: {model.terminaison_saving()}**\n\n")
        model.train_if_needed(config)
        model.sample(config)
        if (not args.no_evaluation):
            full_evaluation(config, model.terminaison_saving())