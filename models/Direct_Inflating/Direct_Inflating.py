import pandas as pd
import numpy as np
import os

from utils.utils_sample import preprocessing_cat_data_dataframe_sampling


def train_sample_Direct_Inflating(args):
    '''
    Replicate the training population at the desired size
    '''

    print("\nSampling by Direct Replication of the training data\n\n")
    
    ##################
    ### Parameters ###
    ##################

    term = "_Direct_Inflating"
    datapath = "Data"
    dataname = args.dataname
    filename_training = args.filename_training
    infoname = args.infoname

    attr_setname = args.attributes_setname
    n_sample = args.n_generation

    info_path = f'{datapath}/{dataname}/{infoname}'
    
    dataset_path = f'{datapath}/{dataname}/{filename_training}'
        
    filename_sampling = (args.sampling_terminaison+"_"+str(n_sample)+term+".").join(filename_training.split('.'))

    folder_sampling = f'{args.sample_folder}/{args.folder_save+term}'
    sampling_file = f'{folder_sampling}/{filename_sampling}'
    
    if (not os.path.exists(folder_sampling)):
        os.makedirs(folder_sampling)

    #################
    ### Load Data ###
    #################

    info = pd.read_csv(info_path, sep = ";")
    info = info[info[attr_setname]][["Type", "Variable_name", "Bin_size"]]
    
    name_cat = info[info["Type"].isin(["binary","category","bool"])].reset_index()["Variable_name"]
    
    df_data = pd.read_csv(dataset_path, sep=";", low_memory=False)[info["Variable_name"]].astype(str)
    
    df_data, _ = preprocessing_cat_data_dataframe_sampling(df_data, args.transform.cat_min_count, name_cat)

    ###################
    ### Replication ###
    ###################
    
    n_replication = np.floor(n_sample/len(df_data)).astype(int)
    n_remaining = n_sample - n_replication*len(df_data)
    
    list_df = [df_data for _ in range (n_replication)]
    list_df.append(df_data.sample(n_remaining))
    
    df_final = pd.concat(list_df, ignore_index=True)

    df_final.to_csv(sampling_file,sep=";", index=False)    