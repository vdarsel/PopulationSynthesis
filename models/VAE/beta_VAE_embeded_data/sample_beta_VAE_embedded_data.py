import torch
import numpy as np
from tqdm import tqdm

from utils.utils_train import get_input_embedded_training_data
from models.VAE.beta_VAE_embeded_data.model_beta_VAE_embedded_data import Beta_VAE_Embedded_data

from utils.utils_dir import get_encoded_filename
from utils.utils_import import import_torch_model

def sample_beta_VAE_from_embedded_data(args, beta, term, k=0): 
    
    print("\n\nSampling beta-VAE model on embedded data\n\n")

    ##################
    ### Parameters ###
    ##################

    device = args.device
        
    n_batch = args.batch_size_generation

    #################
    ### Load Data ###
    #################

    train_z = get_input_embedded_training_data(args)[0]

    mean_train, std = train_z.mean(0), train_z.std(0)
    
    ####################
    ### Import Model ###
    ####################

    import_model = import_torch_model(args, "model", term)
    
    filename_encoded_data = get_encoded_filename(args, term)

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

    np.save(filename_encoded_data, res)
