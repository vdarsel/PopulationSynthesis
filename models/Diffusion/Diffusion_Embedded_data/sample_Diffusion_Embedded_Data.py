import torch
import numpy as np
from utils.utils_train import get_input_embedded_training_data
from models.Diffusion.Diffusion_Embedded_data.model_Diffusion import MLPDiffusion, Model_Diffusion
from tqdm import tqdm

from utils.utils_train import get_input_embedded_training_data
from utils.utils_diffusion import sample
from tqdm import tqdm


def sample_Diffusion_from_embedded_data(args, projection_dim = 1024):
    
    print("\n\nSampling Diffusion model on embedded data\n\n")

    ##################
    ### Parameters ###
    ##################
    
    term = f"_diffusion_{projection_dim}"
        
    name_model = f"model_diffusion_{projection_dim}"

    device = args.device

    path_model = f'ckpt/{args.folder_save}'
    n_batch = args.batch_size_generation
        
    n = int(np.ceil(args.n_generation/n_batch))

    if ("num_steps" in args.__dict__.keys()):
        num_steps = args.num_steps
    else:
        num_steps = 50

    #################
    ### Load Data ###
    #################

    train_z = get_input_embedded_training_data(args)[0]

    in_dim = train_z.shape[1] 

    mean_train, std = train_z.mean(0), train_z.std(0)

    ####################
    ### Import Model ###
    ####################

    import_model = torch.load(f'{path_model}/{name_model}.pt')

    denoise_fn = MLPDiffusion(in_dim, projection_dim).to(device)

    model = Model_Diffusion(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    model.load_state_dict(import_model)

    model.to(device)

    model.eval()
    
    ################
    ### Sampling ###
    ################

    tot= 0
    res = []
    for _ in tqdm(range(n)):
        n_samples = min(args.n_generation-tot, n_batch)
        x = sample(model.denoise_fn_D, n_samples, in_dim, num_steps)
        tot+= n_samples
        res.append((x.cpu()*2 +mean_train).numpy())
    res = np.concatenate(res)
        
    #########################
    ### Save embeded data ###
    #########################

    np.save(f'ckpt/{args.folder_save}/encoded_generated{term}.npy', res)