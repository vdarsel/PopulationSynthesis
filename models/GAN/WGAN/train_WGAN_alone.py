from models.GAN.WGAN.model_WGAN_alone import Generator_Tabular, Discriminator_Tabular, calc_gradient_penalty
import os
from utils.utils_train import set_requires_grad, transform_to_numeric_values
from utils.utils_sample import preprocessing_cat_data_dataframe_sampling, process_nans
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import torch.nn.functional as F

def train_WGAN_alone(args, k=256):
    
    print("\n\nTraining WGAN on raw data\n\n")

    ##################
    ### Parameters ###
    ##################

    datapath = "Data"
    dataname = args.dataname
    filename = args.filename
    infoname = args.infoname
    save_folder = args.folder_save
    attr_setname = args.attributes_setname
    
    info_path = f'{datapath}/{dataname}/{infoname}'

    data_dir = f'{datapath}/{dataname}'
    device = args.device
    
    path_model_save = f'ckpt/{save_folder}'

    path_time = f'ckpt/{args.folder_save}/training_time_WGAN.txt'

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

    info = pd.read_csv(info_path, sep = ";")
    info = info[info[attr_setname]][["Type", "Variable_name"]]

    name_cat = info["Variable_name"][info["Type"].isin(["binary","cat", "bool"])].to_list()
    
    idx_cat = np.arange(len(info))[info["Type"].isin(["binary","cat", "bool"])]
    idx_num = np.arange(len(info))[info["Type"].isin(["int","cont"])]
    
    training_file = f"{data_dir}/{filename}"
    
    training_data = pd.read_csv(training_file, sep = ";", index_col="Original_index")[info["Variable_name"]]
    for idx in name_cat:
        training_data[idx] = training_data[idx].astype(str)
    
    training_data,_ = preprocessing_cat_data_dataframe_sampling(training_data, T_dict['cat_min_count'], name_cat)

    training_data = process_nans(training_data.to_numpy(), idx_num, idx_cat, T_dict['num_nan_policy'], T_dict['cat_nan_policy'])
    X_num, X_cat, categories = transform_to_numeric_values(training_data, idx_num, idx_cat)
    
    X_train_num = torch.Tensor(X_num)
    X_train_cat = torch.LongTensor(X_cat)
    
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
    
    ##################
    ### Save Model ###
    ##################

    if (not os.path.isdir(path_model_save)):
        os.makedirs(path_model_save)
        
    torch.save(generator_net.state_dict(),f'{path_model_save}/generator_WGAN_{k}.pt')
    torch.save(discriminator_net.state_dict(),f'{path_model_save}/discriminator_WGAN_{k}.pt')

    end_time = time.time()
    message = 'Training time: {:.4f} mins'.format((end_time - start_time)/60)
    with open(path_time, "w") as f:
        f.write(message)