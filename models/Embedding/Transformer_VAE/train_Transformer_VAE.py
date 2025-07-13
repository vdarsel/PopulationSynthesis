import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings

import os
from tqdm import tqdm
import time

from models.Embedding.Transformer_VAE.model_Transformer_VAE import Model_VAE, Encoder_model, Decoder_model
from utils.utils_train import preprocess, TabularDataset

warnings.filterwarnings('ignore')




def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def learn_encoding_Transformer_VAE(args):
    
    print("\n\nTraining Transformer VAE for data embedding\n\n")
    
    ##################
    ### Parameters ###
    ##################
    
    datapath = "Data"
    dataname = args.dataname
    filename = args.filename
    infoname = args.infoname
    save_folder = args.folder_save
    attr_setname = args.attributes_setname

    data_dir = f'{datapath}/{dataname}'

    max_beta = args.Transformer_VAE.max_beta
    min_beta = args.Transformer_VAE.min_beta
    lambd = args.Transformer_VAE.lambd

    LR = args.Transformer_VAE.learning_rate
    WD = args.Transformer_VAE.weight_decay_coef
    D_TOKEN = args.Transformer_VAE.dimension_token

    N_HEAD = args.Transformer_VAE.number_heads
    FACTOR = args.Transformer_VAE.factor
    NUM_LAYERS = args.Transformer_VAE.n_layers

    T_dict = {}

    T_dict['normalization'] = args.transform.num_normalization
    T_dict['num_nan_policy'] = args.transform.num_nan_policy
    T_dict['cat_nan_policy'] =  args.transform.cat_nan_policy
    T_dict['cat_min_frequency'] = args.transform.cat_min_frequency
    T_dict['cat_min_count'] = args.transform.cat_min_count
    T_dict['cat_encoding'] = args.transform.cat_encoding


    device =  args.device

    info_path = f'{datapath}/{dataname}/{infoname}'


    ####################
    ### Data loading ###
    ####################

    info = pd.read_csv(info_path, sep = ";")
    info = info[info[attr_setname]][["Type", "Variable_name"]]

    idx_cat = info["Variable_name"][info["Type"].isin(["binary","cat", "bool", "category"])].to_list()
    idx_num = info["Variable_name"][info["Type"].isin(["int","cont", "int64","float64"])].to_list()


    # curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'ckpt/{save_folder}' 

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'
    path_time = f'{ckpt_dir}/training_time_VAE.txt'
    
    X_num, X_cat, categories, d_numerical = preprocess(data_dir, filename, idx_cat, idx_num, T_dict, )

    X_train_num, _ = X_num
    X_train_cat, _ = X_cat

    X_train_num, X_validation_num = X_num
    X_train_cat, X_validation_cat = X_cat


    X_train_num, X_validation_num = torch.tensor(X_train_num).float(), torch.tensor(X_validation_num).float()
    X_train_cat, X_validation_cat =  torch.tensor(X_train_cat), torch.tensor(X_validation_cat)

    train_data = TabularDataset(X_train_num.float(), X_train_cat)

    X_validation_num = X_validation_num.float().to(device)
    X_validation_cat = X_validation_cat.to(device)
    
    batch_size = args.Transformer_VAE.batch_size
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    ##########################
    ### Model construction ###
    ##########################

    model = Model_VAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, bias = True)
    model = model.to(device)

    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
    pre_decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)


    pre_encoder.eval()
    pre_decoder.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters", num_params)

    ######################
    ### Model training ###
    ######################
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)

    num_epochs = args.Transformer_VAE.num_epochs
    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = max_beta
    start_time = time.time()

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0
        curr_acc = 0

        curr_count = 0

        for batch_num, batch_cat in pbar:
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
        
            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl    += loss_kld.item() * batch_length
            curr_acc += batch_size

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count
        train_acc = curr_acc / curr_count
        

        ########################
        ### Model evaluation ###
        ########################

        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_validation_num, X_validation_cat)

            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(X_validation_num, X_validation_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
            val_loss = val_mse_loss.item() + val_ce_loss.item() + beta * val_kl_loss.item()
            if(len(idx_num)==0):
                val_loss = val_ce_loss.item() + beta * val_kl_loss.item()

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")
                
            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10: 
                    beta = beta * lambd
                    if beta < min_beta:
                        break


        print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc, val_acc.item() ))

    end_time = time.time()
    message = 'Training time: {:.4f} mins'.format((end_time - start_time)/60)
    with open(path_time, "w") as f:
        f.write(message)
    
    
    ###################################
    ### Generate and save embedding ###
    ###################################
    
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        print('Successfully load and save the model!')

        train_z = []

        pbar = tqdm(train_loader, total=len(train_loader))
        for batch_num, batch_cat in pbar:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            train_z_temp = pre_encoder(batch_num, batch_cat).detach().cpu().numpy()
            train_z.append(train_z_temp)
        train_z = np.concatenate(train_z)
        X_validation_num = X_validation_num.to(device)
        X_validation_cat = X_validation_cat.to(device)
        validation_z = pre_encoder(X_validation_num, X_validation_cat).detach().cpu().numpy()

        np.save(f'{ckpt_dir}/train_z.npy', train_z)
        np.save(f'{ckpt_dir}/validation_z.npy', validation_z)

        print('Successfully save pretrained embeddings in disk!')