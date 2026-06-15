from models.GAN.WGAN_embeded_data.model_WGAN_embedded import Generator
from utils.utils_train import get_input_embedded_training_data
import torch
from tqdm import tqdm
import numpy as np

from utils.utils_dir import get_encoded_filename
from utils.utils_import import import_torch_model


def sample_WGAN_embedding_data(args, term, k=256):
    
    print("\n\nSampling WGAN model on embedded data\n\n")

    ##################
    ### Parameters ###
    ##################
    
    device = args.device

    train_z, _ = get_input_embedded_training_data(args)
    
    
    #################
    ### Load Data ###
    #################

    in_dim = train_z.shape[1]

    mean, std = train_z.mean(0), train_z.std(0)

    n_batch = args.batch_size_generation
    n = int(np.ceil(args.n_generation/n_batch))
    
    ####################
    ### Import Model ###
    ####################
    
    generator_net = Generator(in_dim,k = k).to(device)

    import_model_generator = import_torch_model(args, "generator", term)
    
    generator_net.load_state_dict(import_model_generator)

    generator_net.to(device)

    generator_net.eval()


    ################
    ### Sampling ###
    ################

    tot= 0
    res = []

    with torch.no_grad():
        for _ in tqdm(range(n)):
            n_samples = min(args.n_generation-tot, n_batch)
            sample_noise = generator_net.sample_latent(n_samples).to(device)
            x = generator_net.features_to_embedding(sample_noise)
            tot+= n_samples
            res.append((x.cpu()*2*std +mean).numpy())
    res = np.concatenate(res)

    #################
    ### Save data ###
    #################

    np.save(get_encoded_filename(args,term), res)