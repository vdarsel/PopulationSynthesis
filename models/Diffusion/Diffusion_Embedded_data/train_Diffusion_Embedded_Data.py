import torch
from torch.utils.data import DataLoader
from models.Diffusion.Diffusion_Embedded_data.model_Diffusion import MLPDiffusion, Model_Diffusion
from utils.utils_train import get_input_embedded_training_data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time

def train_Diffusion_from_embedded_data(args, projection_dim = 1024):

    print("\n\nTraining Diffusion on embedded data\n\n")

    ##################
    ### Parameters ###
    ##################
    
    device = args.device
    
    name_model = f"model_diffusion_{projection_dim}"

    batch_size = args.Diffusion_embedded.batch_size

    num_epochs = args.Diffusion_embedded.n_epochs + 1
    
    #################
    ### Load Data ###
    #################

    train_z,validation_z = get_input_embedded_training_data(args)

    path_model_save = f'ckpt/{args.folder_save}'

    in_dim = train_z.shape[1]

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2 #center data on 0 with the same range
    train_data = train_z
    validation_z = (validation_z - mean) / 2 #center data on 0 with the same range
    validation_data = validation_z
    validation_data = torch.tensor(validation_data).to(device)
    path_time = f'ckpt/{args.folder_save}/training_time_Diffusion_from_embedded_data_{projection_dim}.txt'

    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )
        
    ######################
    ### Initiate Model ###
    ######################
    
    denoise_fn = MLPDiffusion(in_dim, projection_dim).to(device)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model_Diffusion(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

    
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
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

    
        ######################
        ### Evaluate Model ###
        ######################
        
        model.eval()
        with torch.no_grad():
            val_loss = model(validation_data)

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")

            curr_loss = val_loss
            scheduler.step(curr_loss)
            pbar.set_postfix({"Validation Loss": val_loss.item()})

            if curr_loss < best_loss:
                best_loss = loss.item()
                patience = 0
                torch.save(model.state_dict(), f'{path_model_save}/{name_model}.pt')
            else:
                patience += 1
                if patience == args.Diffusion_embedded.patience_max:
                    print('Early stopping')
                    break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f'{path_model_save}/{name_model}_{epoch}.pt')

    end_time = time.time()
    message = 'Training time: {:.4f} mins'.format((end_time - start_time)/60)
    with open(path_time, "w") as f:
        f.write(message)