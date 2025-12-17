import pandas as pd
import os
import numpy as np

from tqdm import tqdm
from time import time

from utils.utils_train import transform_num_into_bins, transform_num_into_quantiles
from utils.utils_sample import preprocessing_cat_data_dataframe_sampling, bins_to_values

def gibbs_sampling_one_pass(data,df_data):
    columns = df_data.columns
    for i in range(len(columns)):
        col = columns[i]
        col_minus_1 = np.delete(columns,i)
        values = df_data[intersection_eq(df_data[col_minus_1],data[col_minus_1]).all(axis=1)][col].values
        np.random.shuffle(values)
        data[col] = values[0]
    return data

def gibbs_sampling_one_pass_np(data_np,data_train_np, n_cols = -1):
    keep_col = np.array([True for _ in range(n_cols)])
    for i in range(n_cols):
        keep_col_temp = keep_col.copy()
        keep_col_temp[i] = False
        values = data_train_np[intersection_eq_np(data_train_np[:,keep_col_temp],data_np[keep_col_temp]),i]
        values = np.reshape(values,-1)
        data_np[i] = values[0]
    return data_np

def gibbs_sampling_one_pass_np_dict_Bayesian(data_np, translation_dict_values, translation_dict_freq, unique_vals, n_cols = -1):
    base_alpha = 0.1 
    keep_col = np.array([True for _ in range(n_cols)])
    for i in range(n_cols):
        keep_col_temp = keep_col.copy() 
        keep_col_temp[i] = False
        code = " ".join(data_np[keep_col_temp])
        if (code in translation_dict_values[i].keys()):
            values = translation_dict_values[i][code]
            freq = translation_dict_freq[i][code]
        else:
            values = unique_vals[i]
            freq = np.zeros(len(values))
        distribution = np.random.dirichlet(alpha=freq+base_alpha)
        data_np[i] = np.random.choice(values,p=distribution)
    return data_np


def intersection_eq(df_: pd.DataFrame, s_: pd.Series) -> pd.DataFrame:
    aligned_df_, aligned_s_ = df_.align(s_, join='inner', axis=1, copy=False)
    return aligned_df_.eq(aligned_s_)

def intersection_eq_np(array_main_: np.ndarray, array_compare_: np.ndarray) -> np.ndarray:
    return (array_main_==array_compare_).all(1)


def MCMC_Bayesian_learn_sample(args):
    
    print("\n\nTraining MCMC with Bayesian posterior distribution\n\n")

    ##################
    ### Parameters ###
    ##################

    term = "_MCMC_Bayesian"
    t0 = time()
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
    sampling_file = f'{folder_sampling}/{filename_sampling.replace(".","_alpha_0_1.")}'
    
    if (not os.path.exists(folder_sampling)):
        os.makedirs(folder_sampling)

    #################
    ### Load Data ###
    #################
    
    info = pd.read_csv(info_path, sep = ";")
    info = info[info[attr_setname]][["Type", "Variable_name", "Bin_size"]]
    
    
    name_cat = info[info["Type"].isin(["binary","category","boolean"])].reset_index()["Variable_name"]
    name_num = info[info["Type"].isin(["int","float"])].reset_index()["Variable_name"]
    
    df_data = pd.read_csv(dataset_path, sep=";", low_memory=False)[info["Variable_name"]].astype(str)
    
    df_data, _ = preprocessing_cat_data_dataframe_sampling(df_data, args.transform.cat_min_count, name_cat)
    
    df_data, dict_translation_inverse = transform_num_into_quantiles(df_data, name_num)
    df_data = transform_num_into_bins(df_data, name_num, args.transform_statistical.bins_num)

    data_tot = df_data.to_numpy().astype(str)
        
    data = data_tot[np.random.randint(len(data_tot))]
    
    res = []
    n_cols = len(data)
    
    ###############################
    ### Preprocess for Sampling ###
    ###############################
    
    print("\nPreprocessing for conditionnal sampling")
    
    transl = []
    
    opposite_data_tot = []
    
    for row in tqdm(data_tot):
        new_row = np.zeros(n_cols, dtype="<U256")
        for i in range(n_cols):
            idx = np.array([True for _ in range(n_cols)])
            idx[i] = False
            new_row[i] = " ".join(row[idx])
        opposite_data_tot.append(new_row)

    opposite_data_tot = np.array(opposite_data_tot, dtype="<U256")
    
    unique_vals = []
        
    for i in tqdm(range(n_cols)):
        vals_opp = (np.unique(opposite_data_tot[:,i]))
        vals = (np.unique(data_tot[:,i]))
        dict_val = {}
        for val in vals:
            dict_val[val] = 0
        dict_trans = {val: dict_val.copy() for val in vals_opp}
        transl.append(dict_trans)
        unique_vals.append(vals)
    
    for j in tqdm(range(len(data_tot))):
        for i in range(n_cols):
            transl[i][opposite_data_tot[j][i]][data_tot[j][i]]+=1
    transl_values = []
    transl_count = []
    
    for i in tqdm(range(n_cols)):
        dict_val = {}
        dict_count = {}
        for key_1 in transl[i]:
            dict_val[key_1] = []
            dict_count[key_1] = []
            for key_2 in transl[i][key_1]:
                dict_val[key_1].append(key_2)
                dict_count[key_1].append(transl[i][key_1][key_2])
            dict_val[key_1] = np.array(dict_val[key_1])
            dict_count[key_1] = np.array(dict_count[key_1])
        transl_values.append(dict_val)
        transl_count.append(dict_count)
    
    print("Preprocessing over")


    ################
    ### Sampling ###
    ################
    
    print("\nSampling")
    for i in tqdm(range(args.MCMC_Bayesian.warm_up+(args.n_generation)*args.MCMC_Bayesian.thinning)):
        data = gibbs_sampling_one_pass_np_dict_Bayesian(data,transl_values, transl_count, unique_vals, n_cols)
        if (i>args.MCMC_Bayesian.warm_up):
            if(i%args.MCMC_Bayesian.thinning==0):
                res.append(data.copy())
                
    df_sample = pd.DataFrame(res, columns=info["Variable_name"])

    #############################
    ### Reverse Preprocessing ###
    #############################

    for name in name_num:
        df_sample[name] = df_sample[name].astype(float)
    
    df_sample = bins_to_values(df_sample, df_data, dict_translation_inverse, name_num)    

    #################
    ### Save data ###
    #################
    
    df_sample.to_csv(sampling_file,sep=";", index=False)

    path_time = f'{folder_sampling}/time.txt'
    message = 'Training time: {:.4f} mins'.format((time()-t0)/60)
    with open(path_time, "w") as f:
        f.write(message)