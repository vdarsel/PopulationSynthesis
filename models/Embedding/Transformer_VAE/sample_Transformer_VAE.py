import numpy as np
import torch
import pandas as pd

import warnings
import os

import src
import shutil

from models.Embedding.Transformer_VAE.model_Transformer_VAE import Decoder_model
from utils.utils_sample import process_nans, get_normalizer_num, get_categories_inverse, preprocessing_cat_data_dataframe_sampling

from utils.utils_dir import get_data_dir, get_folder_sampling, get_file_path_sampling,  get_encoded_filename
from utils.utils_import import get_info_file, import_data, import_torch_model


warnings.filterwarnings('ignore')

def decode_Transformer_VAE(args, term):
    
    name_experience= term.replace("_"," ")
    print(f"\n\n Decoding data for {name_experience} \n\n")
    
    ##################
    ### Parameters ###
    ##################

    filename_training = args.filename_training

    T_dict = {}

    T_dict['normalization'] = args.transform.num_normalization
    T_dict['num_nan_policy'] = args.transform.num_nan_policy
    T_dict['cat_nan_policy'] =  args.transform.cat_nan_policy
    T_dict['cat_min_frequency'] = args.transform.cat_min_frequency
    T_dict['cat_encoding'] = args.transform.cat_encoding

    T = src.Transformations(**T_dict)
    data_dir = get_data_dir(args)
    training_file = os.path.join(data_dir, filename_training)
    folder_sampling = get_folder_sampling(args, term)
    
    encoded_filename = get_encoded_filename(args, term)
    
    min_size_category = args.transform.cat_min_count

    D_TOKEN = args.Transformer_VAE.dimension_token
    
    N_HEAD = args.Transformer_VAE.number_heads
    FACTOR = args.Transformer_VAE.factor
    NUM_LAYERS = args.Transformer_VAE.n_layers

    device =  args.device


    ####################
    ### Loading data ###
    ####################
        
    data_column = np.load(encoded_filename)

    n_sample = data_column.shape[0]

    info = get_info_file(args)[["Type", "Variable_name"]]

    idx_cat = np.arange(len(info))[info["Type"].isin(["binary","cat", "bool", "category"])]
    idx_num = np.arange(len(info))[info["Type"].isin(["int","cont", "int64","float64"])]
    name_cat = info["Variable_name"][info["Type"].isin(["binary","cat", "bool", "category"])]
    
    training_data = import_data(training_file, info["Variable_name"], name_cat)

    ####################
    ### Process data ###
    ####################

    sampling_file = get_file_path_sampling(args, term)

    training_data, _ = preprocessing_cat_data_dataframe_sampling(training_data, min_size_category, name_cat)
    training_data = process_nans(training_data.to_numpy(), idx_num, idx_cat, T_dict['num_nan_policy'], T_dict['cat_nan_policy'])
    if (len(idx_num)>0):
        num_inverse = get_normalizer_num(training_data[:,idx_num], T_dict['normalization'])
    else:
        num_inverse = lambda x: x
    cat_inverse = get_categories_inverse(training_data[:,idx_cat])


    d_numerical = len(idx_num)
    categories = [len(np.unique(training_data[:,id])) for id in idx_cat]


    ##################
    ### Load model ###
    ##################

    decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR)

    decoder.load_state_dict(import_torch_model(args, "decoder_Transformer_VAE", ""))
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
    shutil.copyfile(os.path.join("conf","conf_variable",args.variable+".yml"), os.path.join(folder_sampling, args.variable + ".yml"))
    shutil.copyfile(os.path.join("conf","conf_size",args.str_float + "%.yml"), os.path.join(folder_sampling, args.str_float + "%.yml"))
