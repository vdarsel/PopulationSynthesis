import numpy as np
import torch
import pandas as pd

import warnings
import os

import src
import shutil

from models.Embedding.Transformer_VAE.model_Transformer_VAE import Decoder_model
from utils.utils_sample import process_nans, get_normalizer_num, get_categories_inverse, preprocessing_cat_data_dataframe_sampling

warnings.filterwarnings('ignore')

def decode_Transformer_VAE(args, term):
    
    name_experience= term.replace("_"," ")
    print(f"\n\n Decoding data for {name_experience} \n\n")
    
    ##################
    ### Parameters ###
    ##################

    datapath = "Data"
    dataname = args.dataname
    filename_training = args.filename_training
    infoname = args.infoname
    save_folder = args.folder_save
    attr_setname = args.attributes_setname

    T_dict = {}

    T_dict['normalization'] = args.transform.num_normalization
    T_dict['num_nan_policy'] = args.transform.num_nan_policy
    T_dict['cat_nan_policy'] =  args.transform.cat_nan_policy
    T_dict['cat_min_frequency'] = args.transform.cat_min_frequency
    T_dict['cat_encoding'] = args.transform.cat_encoding

    T = src.Transformations(**T_dict)
    data_dir = f'{datapath}/{dataname}'
    training_file = f'{data_dir}/{filename_training}'
    folder_sampling = f'{args.sample_folder}/{args.folder_save+term}'

    if (not os.path.exists(folder_sampling)):
        os.makedirs(folder_sampling)

    min_size_category = args.transform.cat_min_count

    D_TOKEN = args.Transformer_VAE.dimension_token
    
    N_HEAD = args.Transformer_VAE.number_heads
    FACTOR = args.Transformer_VAE.factor
    NUM_LAYERS = args.Transformer_VAE.n_layers

    device =  args.device

    info_path = f'{datapath}/{dataname}/{infoname}'

    ####################
    ### Loading data ###
    ####################
        
    data_column = np.load(f"ckpt/{save_folder}/encoded_generated{term}.npy")

    n_sample = data_column.shape[0]

    info = pd.read_csv(info_path, sep = ";")
    info = info[info[attr_setname]][["Type", "Variable_name"]].reset_index()
    

    idx_cat = np.arange(len(info))[info["Type"].isin(["binary","cat", "bool", "category"])]
    idx_num = np.arange(len(info))[info["Type"].isin(["int","cont", "int64","float64"])]
    name_cat = info["Variable_name"][info["Type"].isin(["binary","cat", "bool", "category"])]
    
    training_data = pd.read_csv(training_file, sep = ";", low_memory=False)[info["Variable_name"]]
    for idx in name_cat:
        training_data[idx] = training_data[idx].astype(str)

    ####################
    ### Process data ###
    ####################

    filename_sampling = (args.sampling_terminaison+"_"+str(n_sample)+term+".").join(filename_training.split('.'))
    sampling_file = f'{folder_sampling}/{filename_sampling}'

    training_data, _ = preprocessing_cat_data_dataframe_sampling(training_data, min_size_category, name_cat)
    training_data = process_nans(training_data.to_numpy(), idx_num, idx_cat, T_dict['num_nan_policy'], T_dict['cat_nan_policy'])
    if (len(idx_num)>0):
        num_inverse = get_normalizer_num(training_data[:,idx_num], T_dict['normalization'])
    else:
        num_inverse = lambda x: x
    cat_inverse = get_categories_inverse(training_data[:,idx_cat])


    d_numerical = len(idx_num)
    categories = [len(np.unique(training_data[:,id])) for id in idx_cat]

    ckpt_dir = f'ckpt/{save_folder}' 

    ##################
    ### Load model ###
    ##################

    decoder_save_path = f'{ckpt_dir}/decoder_Transformer_VAE.pt'

    decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR)

    decoder.load_state_dict(torch.load(decoder_save_path))
    decoder = decoder.to(device)

    data = data_column.reshape(n_sample, -1, D_TOKEN)

    syn_num = np.zeros((n_sample, d_numerical), dtype=float)
    syn_cat = np.zeros((n_sample,len(categories)), dtype=int)

    batch_size = args.batch_size_generation

    ###################
    ### Decode data ###
    ###################

    for i in range(0,n_sample,batch_size):
        norm_output = decoder(torch.tensor(data[i:i+batch_size]).to(device))
        x_hat_num, x_hat_cat = norm_output

        syn_cat_temp = []
        for pred in x_hat_cat:
            syn_cat_temp.append(pred.argmax(dim = -1))

        syn_num_temp = x_hat_num.cpu().detach().numpy()
        syn_cat_temp = torch.stack(syn_cat_temp).t().cpu().numpy()

        syn_cat[i:i+batch_size] = syn_cat_temp.copy()
        syn_num[i:i+batch_size] = syn_num_temp.copy()

    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)

    final_data = np.zeros((data.shape[0],len(info)), dtype=object)

    ###########################
    ### Postprocessing data ###
    ###########################

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
    shutil.copyfile(f"conf/conf_variable/{args.variable}.yml", f"{folder_sampling}/{args.variable}.yml")
    shutil.copyfile(f"conf/conf_size/{args.str_float}%.yml", f"{folder_sampling}/{args.str_float}%.yml")
