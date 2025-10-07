import torch
import numpy as np
from tqdm import tqdm

from utils.utils_train import get_input_embedded_training_data
from models.VAE.beta_VAE_embeded_data.model_beta_VAE_embedded_data import Beta_VAE_Embedded_data

def sample_beta_VAE_from_embedded_data(args, beta, k=0): 
    
    print("\n\nSampling beta-VAE model on embedded data\n\n")

    ##################
    ### Parameters ###
    ##################

    device = args.device
    
    term=f"_{k}"
    
    path_model = f'ckpt/{args.folder_save}'
    
    n_batch = args.batch_size_generation

    #################
    ### Load Data ###
    #################

    train_z = get_input_embedded_training_data(args)[0]

    mean_train, std = train_z.mean(0), train_z.std(0)
    
    ####################
    ### Import Model ###
    ####################

    import_model = torch.load(f'{path_model}/model_vae_embedded_data_{beta}{term}.pt')

    model = Beta_VAE_Embedded_data(train_z.shape[1], beta,k)
    model.load_state_dict(import_model)

    model.to(device)

    ################
    ### Sampling ###
    ################

    model.eval()

    tot= 0
    res = []
    n = int(np.ceil(args.n_generation/n_batch))
    with torch.no_grad():
        for _ in tqdm(range(n)):
            n_samples = min(args.n_generation-tot, n_batch)
            x = model.sample(n_samples, device)
            tot+= n_samples
            res.append((x.cpu()*2 +mean_train).numpy())
        res = np.concatenate(res)

    np.save(f'ckpt/{args.folder_save}/embedded_generated_VAE_beta_{beta}{term}.npy', res)
