import os
import torch
import numpy as np
import pandas as pd
import shutil

from models.VAE.beta_VAE.model_beta_VAE_alone import Beta_VAE
from utils.utils_sample import get_categories_inverse, get_normalizer_num, preprocessing_cat_data_dataframe_sampling, process_nans

from utils.utils_dir import get_data_dir, get_folder_sampling, get_file_path_sampling
from utils.utils_import import get_info_file, import_data, import_torch_model

def sample_beta_VAE_alone(args,beta, term, k=0): 
    
    print("\n\nSampling beta-VAE model on raw data\n\n")

    ##################
    ### Parameters ###
    ##################
    
    n_sample = args.n_generation
    
    filename_training = args.filename_training
    device = args.device
    
    data_dir = get_data_dir(args)
    
    folder_sampling = get_folder_sampling(args, term)
    sampling_file = get_file_path_sampling(args, term)

    T_dict = {}

    T_dict['normalization'] = args.transform.num_normalization
    T_dict['num_nan_policy'] = args.transform.num_nan_policy
    T_dict['cat_nan_policy'] =  args.transform.cat_nan_policy
    T_dict['cat_min_count'] = args.transform.cat_min_count
    T_dict['cat_encoding'] = args.transform.cat_encoding

    n_batch = args.batch_size_generation
    n = int(np.ceil(n_sample/n_batch))
    
    #################
    ### Load Data ###
    #################
    
    info = get_info_file(args)[["Type", "Variable_name"]]


    name_cat = info[info["Type"].isin(["binary","category","bool"])].reset_index()["Variable_name"]
    idx_cat = np.arange(len(info))[info["Type"].isin(["binary","category","bool"])]
    idx_num = np.arange(len(info))[info["Type"].isin(["int","float"])]

    training_data = import_data(f'{data_dir}/{filename_training}', info["Variable_name"], name_cat)

    if len(idx_cat)>0:
        training_data, _ = preprocessing_cat_data_dataframe_sampling(training_data, T_dict['cat_min_count'], name_cat)
    training_data = process_nans(training_data.to_numpy(), idx_num, idx_cat, T_dict['num_nan_policy'], T_dict['cat_nan_policy'])

    if (len(idx_num)>0):
        num_inverse = get_normalizer_num(training_data[:,idx_num], T_dict['normalization'])
    else:
        num_inverse = lambda x: x
    if (len(idx_cat)>0):
        cat_inverse = get_categories_inverse(training_data[:,idx_cat])
    else:
        cat_inverse = lambda x: x
        
    d_numerical = len(idx_num)
    categories = [len(np.unique(training_data[:,id])) for id in idx_cat]
    
    ####################
    ### Import Model ###
    ####################

    model = Beta_VAE(categories, d_numerical, beta, k).to(device)
    
    import_model = import_torch_model(args, "model", term) 

    model.load_state_dict(import_model)
    
    model.to(device)

    ################
    ### Sampling ###
    ################

    model.eval()

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
    if (not os.path.isdir(folder_sampling)):
        os.makedirs(folder_sampling)
    
    #################
    ### Save data ###
    #################
    
    (pd.DataFrame(final_data, columns=info["Variable_name"]).to_csv(sampling_file,sep=";", index=False))
    shutil.copyfile(f"conf/conf_variable/{args.variable}.yml", f"{folder_sampling}/{args.variable}.yml")
    shutil.copyfile(f"conf/conf_size/{args.str_float}%.yml", f"{folder_sampling}/{args.str_float}%.yml")