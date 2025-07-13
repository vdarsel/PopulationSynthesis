from models.VAE.beta_VAE_embeded_data.model_beta_VAE_embedded_data import Beta_VAE_Embedded_data
from utils.utils_train import get_input_embedded_training_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def train_beta_VAE_from_embedded_data(args,k):

    print("\n\nTraining beta-VAE on embedded data\n\n")

    ##################
    ### Parameters ###
    ##################

    device = args.device
    
    beta = args.beta
    if beta%1==0:
        beta_str = str(beta).split(".0")[0]
    else:
        beta_str = str(beta)

    num_epochs = args.beta_VAE_embedded_data.n_epochs + 1
    batch_size = args.beta_VAE_embedded_data.batch_size

    #################
    ### Load Data ###
    #################

    train_z,validation_z = get_input_embedded_training_data(args)


    path_model_save = f'ckpt/{args.folder_save}'

    in_dim = train_z.shape[1]

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2 #center dat on 0 with the same range
    train_data = train_z
    validation_z = (validation_z - mean) / 2 #center dat on 0 with the same range
    validation_data = validation_z
    validation_data = torch.tensor(validation_data).to(device)
    path_time = f'ckpt/{args.folder_save}/training_time_beta_VAE_{beta_str}_{k}.txt'

    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    
    ######################
    ### Initiate Model ###
    ######################
    
    model = Beta_VAE_Embedded_data(in_dim, beta, k).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters", num_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20)
    
    ###################
    ### Train Model ###
    ###################
    
    model.train()
    torch.save(model.state_dict(), f'{path_model_save}/model_vae_embedded_data_{beta_str}_{k}.pt')

    best_loss = float('inf')
    current_lr = optimizer.param_groups[0]['lr']
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            x_recon, mu, logvar = model(inputs)
        
            loss = model.loss(x_recon, inputs, mu, logvar)

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"":f"Loss: {loss.item()}, MSE:  {F.mse_loss(x_recon, inputs, reduction='sum')}, KL: {-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())}"})
    
        ######################
        ### Evaluate Model ###
        ######################

        model.eval()
        with torch.no_grad():
            x_recon_val, mu_val, logvar_val = model(validation_data)
            val_loss = model.loss(x_recon_val, validation_data, mu_val, logvar_val)

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
                torch.save(model.state_dict(), f'{path_model_save}/model_vae_embedded_data_{beta_str}_{k}.pt')
            else:
                patience += 1
                if patience == args.beta_VAE_embedded_data.patience_max:
                    print('Early stopping')
                    break 
            print(f"Validation Loss: {val_loss.item()}, MSE:  {F.mse_loss(x_recon_val, validation_data, reduction='sum')}, KL: {-0.5 * torch.sum(1 + logvar_val - mu_val.pow(2) - logvar_val.exp())}")
    
    ##################
    ### Save Model ###
    ##################

    end_time = time.time()
    message = 'Training time: {:.4f} mins'.format((end_time - start_time)/60)
    with open(path_time, "w") as f:
        f.write(message)