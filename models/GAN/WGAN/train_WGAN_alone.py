from models.GAN.WGAN.model_WGAN_alone import Generator_Tabular, Discriminator_Tabular, calc_gradient_penalty
import os
from utils.utils_train import set_requires_grad, transform_to_numeric_values, preprocess
from utils.utils_sample import preprocessing_cat_data_dataframe_sampling, process_nans
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import torch.nn.functional as F

from utils.utils_dir import get_data_dir, get_model_torch_path
from utils.utils_import import get_info_file, import_data
from utils.utils_time import save_time

def train_WGAN_alone(args, term, k=256):
    
    print("\n\nTraining WGAN on raw data\n\n")

    ##################
    ### Parameters ###
    ##################

    filename_training = args.filename_training

    data_dir = get_data_dir(args)
    
    device = args.device

    path_generator_save = get_model_torch_path(args, "generator", term)
    path_discriminator_save = get_model_torch_path(args, "discriminator", term)


    T_dict = {}

    T_dict['normalization'] = args.transform.num_normalization
    T_dict['num_nan_policy'] = args.transform.num_nan_policy
    T_dict['cat_nan_policy'] =  args.transform.cat_nan_policy
    T_dict['cat_min_frequency'] = args.transform.cat_min_frequency
    T_dict['cat_min_count'] = args.transform.cat_min_count
    T_dict['cat_encoding'] = args.transform.cat_encoding


    num_epochs = args.WGAN.n_epochs + 1
    batch_size = args.WGAN.batch_size
    
    #################
    ### Load Data ###
    #################

    info = get_info_file(args)[["Type", "Variable_name"]]

    name_cat = info["Variable_name"][info["Type"].isin(["binary","category", "bool"])].to_list()
    name_num= info["Variable_name"][info["Type"].isin(["int","float"])].to_list()
    
    idx_num = np.arange(len(info))[info["Type"].isin(["int","float"])]
    
        
    X_num, X_cat, categories, d_numerical = preprocess(data_dir, filename_training, name_cat, name_num, T_dict, ) 
    # If num_normalization= "quantile", the output distribution is a normal distribution based on the quantile distribution 
    
    X_train_num = torch.Tensor(np.concat(X_num,0))
    X_train_cat = torch.LongTensor(np.concat(X_cat,0))
    
    d_numerical = len(idx_num)
    
    X_train_cat_tab = []
    for i,cat in enumerate(categories):        
        X_train_cat_tab.append(F.one_hot(X_train_cat[:,i], num_classes=cat))
    
    X_train_cat_one_hot = torch.concat(X_train_cat_tab,1)
    
    train_data = torch.concat([X_train_cat_one_hot.float(), X_train_num.float()],1)
    
    ######################
    ### Initiate Model ###
    ######################

    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )
    num_epochs = args.WGAN.n_epochs
    num_epochs_D = args.WGAN.n_epochs_Discriminator
    
    generator_net = Generator_Tabular(categories,d_numerical, k).to(device)
    discriminator_net = Discriminator_Tabular(categories,d_numerical, k).to(device)

    num_params_1 = sum(p.numel() for p in generator_net.parameters())
    num_params_2 = sum(p.numel() for p in discriminator_net.parameters())
    print("Number of parameters", num_params_1+num_params_2)


    optimizer_G = torch.optim.Adam(generator_net.parameters(), lr=5e-4, weight_decay=0)
    optimizer_D = torch.optim.Adam(discriminator_net.parameters(), lr=1e-3, weight_decay=0)
    
    ###################
    ### Train Model ###
    ###################

    generator_net.train()

    start_time = time.time()
    generator_net.train()
    discriminator_net.train()

    loss_G_epoch = []
    loss_D_real_epoch = []
    loss_D_fake_epoch = []  
    i=0

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        
        ## training phase
        loss_G_train = []
        loss_D_real_train = []
        loss_D_fake_train = []
        loss_D_train = []

        for batch_data in pbar:
            i+=1
            batch_size_data = len(batch_data)

            real_data = batch_data.to(device)
            
            input = generator_net.sample_latent(batch_size_data).to(device)
            fake_data_num, fake_data_cat = generator_net(input)
            
            fake_data = torch.concat([fake_data_cat,fake_data_num],1)
            
            # backward discriminator_net
            set_requires_grad(discriminator_net, True)
            optimizer_D.zero_grad()

            pred_real = discriminator_net(real_data)
            pred_fake = discriminator_net(fake_data.detach())
            
            fake_data_num = fake_data_num.detach()
            

            # GAN Discriminator Loss
            loss_D_real = (torch.mean(pred_real))
            loss_D_fake = (torch.mean(pred_fake))

            # Gradient penalty Loss
            GP_loss = calc_gradient_penalty(discriminator_net, real_data, fake_data, device, lambda_=1)

            # GAN Discriminator Loss
            loss_D = (loss_D_fake - loss_D_real) + GP_loss

            loss_D.backward()
            optimizer_D.step()

            if ((i%args.WGAN.freq_train_G == 0)&(epoch>num_epochs_D)):
                set_requires_grad(discriminator_net, False)
                optimizer_G.zero_grad()
                
                input = generator_net.sample_latent(batch_size_data).to(device)
                fake_data_num, fake_data_cat = generator_net(input)
                fake_data = torch.concat([fake_data_cat,fake_data_num],1)

                pred_fake = discriminator_net(fake_data)

                loss_G = -torch.mean(pred_fake)

                loss_G.backward()
                optimizer_G.step()

                loss_G_train += [loss_G.item()]
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]
            loss_D_train += [loss_D.item()]

        print('epoch: {}, Train loss Discriminator fake data: {:.6f}, Train loss Discriminator real data: {:.6f}, Train loss Discriminator : {:.6f}, Train loss Generator:{:.6f}'.format(epoch, np.mean(loss_D_fake_train), np.mean(loss_D_real_train), np.mean(loss_D_train) ,np.mean(loss_G_train) ))
        loss_G_epoch.append(np.mean(loss_G_train))
        loss_D_real_epoch.append(np.mean(loss_D_real_train))
        loss_D_fake_epoch.append(np.mean(loss_D_fake_train))
        
        # Save checkpoint every 100 epochs
        if epoch % 100 == 0:
            torch.save(generator_net.state_dict(), path_generator_save)

    ##################
    ### Save Model ###
    ##################

    torch.save(generator_net.state_dict(),path_generator_save)
    torch.save(discriminator_net.state_dict(), path_discriminator_save)

    save_time(start_time, args, term)