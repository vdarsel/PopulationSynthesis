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

from utils.utils_train import generate_train_validation_set
from utils.utils_sample import preprocessing_cat_data_dataframe_sampling
from utils.utils_CTGAN.data_transformer import DataTransformer
from models.VAE.TVAE.model_TVAE import Encoder_TVAE, Decoder_TVAE, loss_function_TVAE

from utils.utils_dir import get_data_dir, get_ckpt_dir, get_model_torch_path
from utils.utils_import import get_info_file, import_data
from utils.utils_time import save_time

warnings.filterwarnings('ignore')


def train_TVAE(args, beta, term, dim=256):

    print("\n\nTraining TVAE on raw data\n\n")

    ##################
    ### Parameters ###
    ##################

    beta_str = beta
    beta = float(beta)
    
    filename_training = args.filename_training

    data_dir = get_data_dir(args)

    device =  args.device

    info = get_info_file(args)[["Type", "Variable_name"]]

    columns = info["Variable_name"]
    name_cat = info["Variable_name"][(info["Type"].isin(["binary","cat","bool","category"]))].to_list()

    ckpt_dir = get_ckpt_dir(args)
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    encoder_save_path = get_model_torch_path(args, "encoder", term)
    decoder_save_path = get_model_torch_path(args, "decoder", term)
    training_data_save_path = os.path.join(ckpt_dir, "TVAE_" + filename_training + "_train_train_dim_" + dim + "_beta_" + beta_str + ".csv")

    path_time = os.path.join(ckpt_dir, "training_time_TVAE.txt")

    ###################
    ### Import Data ###
    ###################

    dataset_train = import_data(os.path.join(data_dir, filename_training), columns, name_cat)
        
    idx_training, idx_validation = generate_train_validation_set(dataset_train[name_cat])

    min_size_category = args.transform.cat_min_count

    dataset_train, dfs  = preprocessing_cat_data_dataframe_sampling(dataset_train, min_size_category, name_cat)
    
    for idx in name_cat:
        dataset_train[idx] = dataset_train[idx].astype("category")

    dataset_train_train, dataset_train_validation = dataset_train.iloc[idx_training],dataset_train.iloc[idx_validation]
    
    transformer_data = DataTransformer()
    dataset_train_train_for_saving = dataset_train_train.copy()
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
    
    save_time(start_time, args, term)
    
    torch.save(decoder.state_dict(), decoder_save_path)
    torch.save(encoder.state_dict(), encoder_save_path)
    dataset_train_train_for_saving.to_csv(training_data_save_path, sep=";", index=False)