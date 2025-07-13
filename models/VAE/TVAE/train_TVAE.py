import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import warnings

import os
from tqdm import tqdm
import time

from utils.utils_train import TabularDataset
from utils.utils_sample import preprocessing_cat_data_dataframe_sampling
from utils.utils_CTGAN.data_transformer import DataTransformer
from models.VAE.TVAE.model_TVAE import Encoder_TVAE, Decoder_TVAE, loss_function_TVAE

warnings.filterwarnings('ignore')


def train_TVAE(args, beta, dim=256):

    print("\n\nTraining TVAE on raw data\n\n")

    ##################
    ### Parameters ###
    ##################

    beta_str = beta
    beta = float(beta)
    
    datapath = "Data"
    dataname = args.dataname
    filename = args.filename
    infoname = args.infoname
    save_folder = args.folder_save
    attr_setname = args.attributes_setname

    data_dir = f'{datapath}/{dataname}'


    device =  args.device

    info_path = f'{datapath}/{dataname}/{infoname}'

    info = pd.read_csv(info_path, sep = ";")
    info = info[info[attr_setname]][["Type", "Variable_name"]]

    columns = info["Variable_name"]
    name_cat = info["Variable_name"][(info["Type"].isin(["binary","cat","bool","category"]))].to_list()

    ckpt_dir = f'ckpt/{save_folder}' 
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    encoder_save_path = f'{ckpt_dir}/encoder_TVAE_dim_{dim}_beta_{beta_str}.pt'
    decoder_save_path = f'{ckpt_dir}/decoder_TVAE_dim_{dim}_beta_{beta_str}.pt'
    path_time = f'{ckpt_dir}/training_time_TVAE.txt'

    ###################
    ### Import Data ###
    ###################

    dataset_train = pd.read_csv(f"{data_dir}/{filename}", sep=";", index_col="Original_index", low_memory=False)[info["Variable_name"]]
    dataset_train_train = pd.read_csv(f"{data_dir}/{filename.replace("_train.csv","_train_train.csv")}", sep=";", index_col="Original_index", low_memory=False)[info["Variable_name"]]
    dataset_train_validation = pd.read_csv(f"{data_dir}/{filename.replace("_train.csv","_train_validation.csv")}", sep=";", index_col="Original_index", low_memory=False)[info["Variable_name"]]

    dataset_train = dataset_train[columns]
    for idx in name_cat:
        dataset_train[idx] = dataset_train[idx].astype(str)
        dataset_train_train[idx] = dataset_train_train[idx].astype(str)
        dataset_train_validation[idx] = dataset_train_validation[idx].astype(str)

    min_size_category = args.transform.cat_min_count

    dataset_train, dfs  = preprocessing_cat_data_dataframe_sampling(dataset_train, min_size_category, name_cat, [dataset_train_train, dataset_train_validation])
    dataset_train_train, dataset_train_validation = dfs
    
    for idx in name_cat:
        dataset_train[idx] = dataset_train[idx].astype("category")

    transformer_data = DataTransformer()
    transformer_data.fit(dataset_train_train, name_cat)
    dataset_train_train = transformer_data.transform(dataset_train_train)
    dataset_train_validation = transformer_data.transform(dataset_train_validation)

    dataset_train_train = TensorDataset(torch.from_numpy(dataset_train_train.astype('float32')).to(device))
    data_validation = torch.from_numpy(dataset_train_validation.astype('float32')).to(device)
    loader = DataLoader(dataset_train_train, batch_size=args.TVAE.batch_size, shuffle=True, drop_last=False)

    transformed_data_dim = transformer_data.output_dimensions
    
    ######################
    ### Initiate Model ###
    ######################
    
    encoder = Encoder_TVAE(transformed_data_dim, dim).to(device)
    decoder = Decoder_TVAE(dim, transformed_data_dim).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20)

    num_epochs = args.TVAE.n_epochs
    best_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    loss_values_pd = pd.DataFrame(columns=['Epoch', 'Mean Loss', 'Validation Loss'])
    iterator = tqdm(range(num_epochs))
    iterator_description = 'Loss: {loss:.3f}'
    iterator.set_description(iterator_description.format(loss=0))
    start_time = time.time()

    for i in iterator:
        loss_values = []
        batch = []
        for id_, data in enumerate(loader):
            optimizer.zero_grad()
            real = data[0].to(device)
            mu, std, logvar = encoder(real)
            eps = torch.randn_like(std)
            emb = eps * std + mu
            rec, sigmas = decoder(emb)
            loss_1, loss_2 = loss_function_TVAE(
                rec,
                real,
                sigmas,
                mu,
                logvar,
                transformer_data.output_info_list,
            )
            loss = loss_1 + beta * loss_2
            loss.backward()
            optimizer.step()
            decoder.sigma.data.clamp_(0.01, 1.0)

            batch.append(id_)
            loss_values.append(loss.detach().cpu().item())
    
        ######################
        ### Evaluate Model ###
        ######################

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            mu_val, std_val, logvar_val = encoder(data_validation)
            eps_val = torch.randn_like(std_val)
            emb_val = eps_val * std_val + mu_val
            rec_val, sigmas_val = decoder(emb_val)
            loss_1_val, loss_2_val = loss_function_TVAE(
                rec_val,
                data_validation,
                sigmas_val,
                mu_val,
                logvar_val,
                transformer_data.output_info_list,
            )
            val_loss = loss_1_val + beta * loss_2_val

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")

            curr_loss = val_loss
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss.item()
                patience = 0
            else:
                patience += 1
                if patience == args.TVAE.patience_max:
                    print('Early stopping')
                    break 


        epoch_loss_df = pd.DataFrame({
            'Epoch': [i],
            'Mean Loss': [np.mean(loss_values)],
            'Validation Loss': [val_loss.detach().cpu().item()]
        })
        if not loss_values_pd.empty:
            loss_values_pd = pd.concat([loss_values_pd, epoch_loss_df]).reset_index(
                drop=True
            )
        else:
            loss_values_pd = epoch_loss_df

        iterator.set_description(
                iterator_description.format(loss=loss.detach().cpu().item())
            )

    ##################
    ### Save Model ###
    ##################
    
    end_time = time.time()
    message = 'Training time: {:.4f} mins'.format((end_time - start_time)/60)
    with open(path_time, "w") as f:
        f.write(message)

    
    torch.save(decoder.state_dict(), decoder_save_path)
    torch.save(encoder.state_dict(), encoder_save_path)