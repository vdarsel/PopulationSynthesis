from models.GAN.WGAN.model_WGAN_alone import Generator_Tabular
import os

import torch
import numpy as np
from utils.utils_sample import get_categories_inverse, get_normalizer_num, preprocessing_cat_data_dataframe_sampling, process_nans

import shutil
import pandas as pd

def sample_WGAN_alone(args, k = 256):
    
    print("\n\nSampling WGAN model on raw data\n\n")

    ##################
    ### Parameters ###
    ##################
    
    n_sample = args.n_generation
    term = f"_WGAN_no_embed_{k}"
    datapath = "Data"
    dataname = args.dataname
    filename = args.filename
    infoname = args.infoname
    attr_setname = args.attributes_setname
    device = args.device
    data_dir = f'{datapath}/{dataname}'
    
    if "_train" in args.filename:
        filename_training = filename
    else:
        filename_training = '_train.'.join(args.filename.split('.'))
    
    filename_sampling = (args.sampling_terminaison+"_"+str(n_sample)+term+".").join(args.filename.split('.'))
    folder_sampling = f'{args.sample_folder}/{args.folder_save+term}'
    sampling_file = f'{folder_sampling}/{filename_sampling}'

    path_model = f'ckpt/{args.folder_save}'

    T_dict = {}

    T_dict['normalization'] = args.transform.num_normalization
    T_dict['num_nan_policy'] = args.transform.num_nan_policy
    T_dict['cat_nan_policy'] =  args.transform.cat_nan_policy
    T_dict['cat_min_count'] = args.transform.cat_min_count
    T_dict['cat_encoding'] = args.transform.cat_encoding
    
    info_path = f'{datapath}/{dataname}/{infoname}'
    
    training_file = f'{data_dir}/{filename_training}'

    n_batch = args.batch_size_generation
    n = int(np.ceil(n_sample/n_batch))

    if (not os.path.isdir(folder_sampling)):
        os.makedirs(folder_sampling)    
    
    #################
    ### Load Data ###
    #################

    info = pd.read_csv(info_path, sep = ";")
    info = info[info[attr_setname]][["Type", "Variable_name"]].reset_index()

    idx_cat = np.arange(len(info))[info["Type"].isin(["binary","cat","bool"])]
    idx_num = np.arange(len(info))[info["Type"].isin(["int","cont"])]
    name_cat = info["Variable_name"][info["Type"].isin(["binary","cat","bool"])]


    training_data = pd.read_csv(training_file, sep = ";", index_col="Original_index")[info["Variable_name"]]
    for idx in name_cat:
        training_data[idx] = training_data[idx].astype(str)
    
    training_data,_ = preprocessing_cat_data_dataframe_sampling(training_data, T_dict['cat_min_count'], name_cat)
    training_data = process_nans(training_data.to_numpy(), idx_num, idx_cat, T_dict['num_nan_policy'], T_dict['cat_nan_policy'])
    if (len(idx_num)>0):
        num_inverse = get_normalizer_num(training_data[:,idx_num], T_dict['normalization'])
    else:
        num_inverse = lambda x: x
    cat_inverse = get_categories_inverse(training_data[:,idx_cat])

    d_numerical = len(idx_num)
    categories = [len(np.unique(training_data[:,id])) for id in idx_cat]


    ####################
    ### Import Model ###
    ####################

    import_model = torch.load(f'{path_model}/generator_WGAN_{k}.pt')

    model = Generator_Tabular(categories, d_numerical,k).to(device)

    model.load_state_dict(import_model)

    model.to(device)
    
    model.eval()

    ################
    ### Sampling ###
    ################

    tot= 0
    syn_num = np.zeros((n_sample, d_numerical), dtype=float)
    syn_cat = np.zeros((n_sample,len(categories)), dtype=int)

    i=0
    with torch.no_grad():
        for i in range(0,n*n_batch,n_batch):
            n_samples = min(n_sample-tot, n_batch)
            x_hat_cat, x_hat_num = model.sample(n_samples, device)
            syn_cat_temp = []
            for pred in x_hat_cat:
                syn_cat_temp.append(pred.argmax(dim = -1))
            syn_num_temp = x_hat_num.cpu().detach().numpy()
            syn_cat_temp = torch.stack(syn_cat_temp).t().cpu().numpy()

            syn_cat[i:i+n_batch] = syn_cat_temp.copy()
            syn_num[i:i+n_batch] = syn_num_temp.copy()

    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)

    final_data = np.zeros((n_sample,len(info)), dtype=object)

    for i,id_cat in enumerate(idx_cat):
        final_data[:,id_cat] = syn_cat[:,i]
        
    for i,id_num in enumerate(idx_num):
        if(info["Type"][id_num]=="int"):
            final_data[:,id_num] = syn_num[:,i].astype(int)
        else:
            final_data[:,id_num] = syn_num[:,i]
    
    #################
    ### Save data ###
    #################

    (pd.DataFrame(final_data, columns=info["Variable_name"]).to_csv(sampling_file,sep=";", index=False))
    shutil.copyfile(f"../../Configuration/conf_variable/{args.variable}.yml", f"{folder_sampling}/{args.variable}.yml")
    shutil.copyfile(f"../../Configuration/conf_size/{args.str_float}%.yml", f"{folder_sampling}/{args.str_float}%.yml")
