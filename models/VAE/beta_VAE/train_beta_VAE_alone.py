from models.VAE.beta_VAE.model_beta_VAE_alone import Beta_VAE
from utils.utils_train import preprocess, TabularDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd

import warnings

from tqdm import tqdm
import time

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


def train_beta_VAE_alone(args,k=0):

    print("\n\nTraining beta-VAE on raw data\n\n")

    ##################
    ### Parameters ###
    ##################

    datapath = "Data"
    dataname = args.dataname
    filename = args.filename
    infoname = args.infoname
    attr_setname = args.attributes_setname
    info_path = f'{datapath}/{dataname}/{infoname}'

    data_dir = f'{datapath}/{dataname}'
    device = args.device

    path_model_save = f'ckpt/{args.folder_save}'

    beta = args.beta
    if beta%1==0:
        beta_str = str(beta).split(".0")[0]
    else:
        beta_str = str(beta)

    path_time = f'ckpt/{args.folder_save}/training_time_beta_VAE_{beta_str}_{k}.txt'

    T_dict = {}

    T_dict['normalization'] = args.transform.num_normalization
    T_dict['num_nan_policy'] = args.transform.num_nan_policy
    T_dict['cat_nan_policy'] =  args.transform.cat_nan_policy
    T_dict['cat_min_frequency'] = args.transform.cat_min_frequency
    T_dict['cat_min_count'] = args.transform.cat_min_count
    T_dict['cat_encoding'] = args.transform.cat_encoding

    num_epochs = args.beta_VAE.n_epochs + 1
    batch_size = args.beta_VAE.batch_size
    
    #################
    ### Load Data ###
    #################

    info = pd.read_csv(info_path, sep = ";")
    info = info[info[attr_setname]][["Type", "Variable_name"]]

    idx_cat = info["Variable_name"][info["Type"].isin(["binary","cat", "bool"])].to_list()
    idx_num = info["Variable_name"][info["Type"].isin(["int","cont"])].to_list()


    X_num, X_cat, categories, d_numerical = preprocess(data_dir, filename, idx_cat, idx_num, T_dict, )
    n_cat = len(categories)    
    
    X_train_num, X_validation_num = X_num
    X_train_cat, X_validation_cat = X_cat
    
    X_train_num, X_validation_num = (torch.tensor(X_train_num).float()-0.5)*2, (torch.tensor(X_validation_num).float()-0.5)*2
    X_train_cat, X_validation_cat =  torch.tensor(X_train_cat), torch.tensor(X_validation_cat)

    X_train_cat_tab, X_validation_cat_tab = [],[]
    for i,cat in enumerate(categories):
        X_train_cat_tab.append(F.one_hot(X_train_cat[:,i], num_classes=cat))
        X_validation_cat_tab.append(F.one_hot(X_validation_cat[:,i], num_classes=cat))
    
    X_train_cat_one_hot = torch.concat(X_train_cat_tab,1)
    X_validation_cat_one_hot = torch.concat(X_validation_cat_tab,1)
    
    train_data = TabularDataset(X_train_num.float(), X_train_cat_one_hot.float())

    X_validation = torch.concat([X_validation_num,X_validation_cat_one_hot], 1).float().to(device)
    X_validation_num = X_validation_num.float().to(device)
    X_validation_cat_one_hot = X_validation_cat_one_hot.to(device)

    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )
    
    ######################
    ### Initiate Model ###
    ######################

    if (k!=0):
        model = Beta_VAE(categories, d_numerical, beta, k).to(device)
    else:
        model = Beta_VAE(categories, d_numerical, beta).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters", num_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20)
    
    ###################
    ### Train Model ###
    ###################
    
    model.train()

    best_loss = float('inf')
    current_lr = optimizer.param_groups[0]['lr']
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch_num, batch_cat_one_hot in pbar:
            inputs = torch.concat([batch_num,batch_cat_one_hot],1).float().to(device)
            recon_x_cat, recon_x_cont, mu, logvar = model(inputs)
            x_num_batch = batch_num.float().to(device) 
            x_cat_batch = batch_cat_one_hot.float().to(device)
            loss = model.loss(recon_x_cat, recon_x_cont, x_cat_batch, x_num_batch, n_cat, mu, logvar)
            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    
        ######################
        ### Evaluate Model ###
        ######################

        model.eval()
        with torch.no_grad():
            # print(X_validation.shape)
            X_recon_val_cat, x_recon_val_num, mu_val, logvar_val = model(X_validation)
            val_loss = model.loss(X_recon_val_cat, x_recon_val_num, X_validation_cat_one_hot.float(), X_validation_num, n_cat, mu_val, logvar_val)
            # raise
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
                torch.save(model.state_dict(), f'{path_model_save}/model_vae_alone_{beta_str}_{k}.pt')
                # print(epoch)
            else:
                patience += 1
                if patience == args.beta_VAE.patience_max:
                    print('Early stopping')
                    break 
        print('epoch: {}, Train Loss:{:.6f}, Val Loss:{:.6f}'.format(epoch, batch_loss/len_input, val_loss.item()))

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f'{path_model_save}/model_beta_VAE_{epoch}_{beta_str}_{k}.pt')

    
    ##################
    ### Save Model ###
    ##################
    
    torch.save(model.state_dict(), f'{path_model_save}/model_beta_VAE_{beta_str}_{k}.pt')

    end_time = time.time()
    message = 'Training time: {:.4f} mins'.format((end_time - start_time)/60)
    with open(path_time, "w") as f:
        f.write(message)
