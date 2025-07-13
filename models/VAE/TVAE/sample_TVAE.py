import numpy as np
import torch
import pandas as pd

import os
from utils.utils_sample import preprocessing_cat_data_dataframe_sampling
from utils.utils_CTGAN.data_transformer import DataTransformer
from models.VAE.TVAE.model_TVAE import Decoder_TVAE


def sample_TVAE(args, beta, k=256): 
    
    print("\n\nSampling TVAE model on raw data\n\n")

    ##################
    ### Parameters ###
    ##################
    
    device = args.device

    term=f"_TVAE_beta_{beta}_{k}"

    save_folder = args.folder_save
    attr_setname = args.attributes_setname

    datapath = "Data"
    dataname = args.dataname
    filename = args.filename
    infoname = args.infoname

    data_dir = f'{datapath}/{dataname}'

    info_path = f'{datapath}/{dataname}/{infoname}'
    folder_sampling = f'{args.sample_folder}/{args.folder_save+term}'
    filename_sampling = (args.sampling_terminaison+"_"+str(args.n_generation)+term+".").join(args.filename.split('.'))
    sampling_file = f'{folder_sampling}/{filename_sampling}'

    ckpt_dir = f'ckpt/{save_folder}' 
    
    if (not os.path.isdir(folder_sampling)):
        os.makedirs(folder_sampling)


    n_batch = args.batch_size_generation
    steps = args.n_generation // n_batch + 1
    embedding_dim = max(k//8,1)

    #################
    ### Load Data ###
    #################

    info = pd.read_csv(info_path, sep = ";")
    info = info[info[attr_setname]][["Type", "Variable_name"]]

    columns = info["Variable_name"]
    name_cat = info["Variable_name"][(info["Type"].isin(["binary","cat","bool","category"]))].to_list()

    dataset_train = pd.read_csv(f"{data_dir}/{filename}", sep=";", index_col="Original_index", low_memory=False)[info["Variable_name"]]
    dataset_train_train = pd.read_csv(f"{data_dir}/{filename.replace("_train.csv","_train_train.csv")}", sep=";", index_col="Original_index", low_memory=False)[info["Variable_name"]]

    dataset_train = dataset_train[columns]
    for idx in name_cat:
        dataset_train[idx] = dataset_train[idx].astype(str)
        dataset_train_train[idx] = dataset_train_train[idx].astype(str)

    min_size_category = args.transform.cat_min_count

    dataset_train, dfs  = preprocessing_cat_data_dataframe_sampling(dataset_train, min_size_category, name_cat, [dataset_train_train])
    dataset_train_train = dfs[0]

    transformer_data = DataTransformer()
    transformer_data.fit(dataset_train_train, name_cat)

    transformed_data_dim = transformer_data.output_dimensions

    ####################
    ### Import Model ###
    ####################

    import_model = torch.load(f'{ckpt_dir}/decoder_TVAE_dim_{k}_beta_{beta}.pt')

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