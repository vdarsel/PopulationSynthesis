import numpy as np
import torch
import pandas as pd

import os
from utils.utils_sample import preprocessing_cat_data_dataframe_sampling
from utils.utils_CTGAN.data_transformer import DataTransformer
from models.VAE.TVAE.model_TVAE import Decoder_TVAE

from utils.utils_dir import get_data_dir, get_ckpt_dir, get_folder_sampling, get_file_path_sampling
from utils.utils_import import get_info_file, import_data, import_torch_model


def sample_TVAE(args, beta, term, k=256): 
    
    print("\n\nSampling TVAE model on raw data\n\n")

    ##################
    ### Parameters ###
    ##################
    
    device = args.device

    ckpt_dir = get_ckpt_dir(args)

    filename_training = args.filename_training

    train_train_data_save_path = os.path.join(ckpt_dir, "TVAE_" + filename_training + "_train_train_dim_" + k + "_beta_" + beta + ".csv")

    data_dir = get_data_dir(args)

    folder_sampling = get_folder_sampling(args, term)   
    sampling_file = get_file_path_sampling(args, term) 

    
    if (not os.path.isdir(folder_sampling)):
        os.makedirs(folder_sampling)


    n_batch = args.batch_size_generation
    steps = args.n_generation // n_batch + 1
    embedding_dim = max(k//8,1)

    #################
    ### Load Data ###
    #################

    info = get_info_file(args)[["Type", "Variable_name"]]

    columns = info["Variable_name"]
    name_cat = info["Variable_name"][(info["Type"].isin(["binary","bool","category"]))].to_list()

    dataset_train = import_data(os.path.join(data_dir, filename_training), columns, name_cat)
    dataset_train_train = import_data(train_train_data_save_path, columns, name_cat)

    min_size_category = args.transform.cat_min_count

    dataset_train, dfs  = preprocessing_cat_data_dataframe_sampling(dataset_train, min_size_category, name_cat, [dataset_train_train])
    dataset_train_train = dfs[0]

    transformer_data = DataTransformer()
    transformer_data.fit(dataset_train_train, name_cat)

    transformed_data_dim = transformer_data.output_dimensions

    ####################
    ### Import Model ###
    ####################

    import_model = import_torch_model(args, "decoder", term)

    decoder = Decoder_TVAE(k, transformed_data_dim).to(device)
    decoder.load_state_dict(import_model)

    decoder.to(device)

    ################
    ### Sampling ###
    ################

    decoder.eval()

    data = []
    for _ in range(steps):
        mean = torch.zeros(n_batch, embedding_dim)
        std = mean + 1
        noise = torch.normal(mean=mean, std=std).to(device)
        fake, sigmas = decoder(noise)
        fake = torch.tanh(fake)
        data.append(fake.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)
    data = data[:args.n_generation]
    
    data = transformer_data.inverse_transform(data, sigmas.detach().cpu().numpy())

    data.to_csv(f"{sampling_file}", sep=";", index=False)