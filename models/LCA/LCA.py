import matplotlib.pyplot as plt

import pandas as pd
import os
import numpy as np

from time import time

from utils.utils_train import transform_num_into_bins, transform_num_into_quantiles
from utils.utils_sample import preprocessing_cat_data_dataframe_sampling, bins_to_values

from utils.utils_dir import get_data_dir, get_folder_sampling, get_file_path_sampling
from utils.utils_import import get_info_file, import_data

from utils.utils_time import save_time

def data_to_numpy(data):
    res = np.zeros(data.shape, int)
    dict_inverse = []
    for i,col in enumerate(data.columns):
        unique_vals = data[col].unique()
        transf = {a:i for i,a in enumerate(unique_vals)}
        dict_inverse.append({i:a for i,a in enumerate(unique_vals)})
        res[:,i] = data[col].map(transf)
    return res, dict_inverse
        


def find_best_numbers_clusters(filters, pop):
    k_min,k_max = 2, 40
    BIC_values = []
    for n_clusters in range(k_min,k_max):
        lambda_est = np.ones(n_clusters)/n_clusters
        coefs_est = [[init_proba(f.shape[0]) for f in (filters)] for _ in range(n_clusters)]
        
        lambdas_new = lambda_est
        coefs_new = coefs_est

        LLs = [compute_LL(filters, lambdas_new, coefs_new)]
        stop = False
        max_iter = 10000
        iter = 0 
        while not stop:
            iter+=1
            q = q_z_x(lambdas_new, coefs_new, pop)
            lambdas_new = q.sum(0)/q.sum()
            coefs_new = update_coefs(q, filters)
            LLs.append(compute_LL(filters, lambdas_new, coefs_new))
            if((LLs[-1]-LLs[-2]<0.01)|(iter>max_iter)):
                stop = True
                
        k_per_cluster = np.sum([len(c)-1 for c in coefs_est[0]])+1
        k = k_per_cluster*len(lambdas_new)-1
        BIC = k*np.log(len(pop))-2*LLs[-1]
        BIC_values.append(BIC)
        print(BIC, len(LLs),n_clusters)
        if (np.min(BIC_values)<np.min(BIC_values[-10:])):
            break
    return k_min + np.argmin(BIC_values)


def get_LCA_coefs(pop, n_clusters):
    filters = generate_filters(pop)
    lambda_est = np.ones(n_clusters)/n_clusters
    coefs_est = [[init_proba(f.shape[0]) for f in (filters)] for _ in range(n_clusters)]
    
    lambdas_new = lambda_est
    coefs_new = coefs_est

    LLs = [compute_LL(filters, lambdas_new, coefs_new)]
    stop = False
    while not stop:
        q = q_z_x(lambdas_new, coefs_new, pop)
        lambdas_new = q.sum(0)/q.sum()
        coefs_new = update_coefs(q, filters)
        LLs.append(compute_LL(filters, lambdas_new, coefs_new))
        if(LLs[-1]-LLs[-2]<0.01):
            stop = True
    
    return coefs_new, lambdas_new, LLs

def sampling_from_coefs(coefs, lambdas, n_tot):
    padded_coefs = pad_coefs(coefs)
    return sampling_LCA(lambdas, padded_coefs, n_tot)



def generate_coefs_d(d):
    vals = np.random.rand(d)
    return vals/np.sum(vals)

def init_proba(d):
    res = np.random.rand(d)
    res = res/res.sum()
    return res

def q_z_x(lambdas,coefs,X):
    res = []
    padded_coefs = pad_coefs(coefs)
    d_idx  = np.arange(X.shape[1])
    selected = padded_coefs[:, d_idx, X]
    res = (np.prod(selected,2)).T*lambdas
    return res/np.sum(res,1,keepdims=True)

def generate_filters(pop):
    filters = []
    n_values = np.max(pop,0)+1
    for i in range(pop.shape[1]):
        f = []
        for k in range(n_values[i]):
            f.append((pop[:,i]==k).astype(int))
        filters.append(np.array(f))
    return filters

def update_coefs(q_values, filters):
    res = []
    for j in range(q_values.shape[1]):
        res_cluster = []
        for i in range(len(filters)):
            res_variable = []
            for k in range(len(filters[i])):
                res_variable.append(np.sum(q_values[:,j]*filters[i][k]))
            res_variable = np.array(res_variable)
            res_cluster.append(res_variable/np.sum(res_variable, keepdims=True))
        res.append(res_cluster)
    return res

def compute_LL(filters,lambdas,coefs):
    res = np.zeros(len(filters[0][0]))
    for j in range(len(lambdas)):
        res_cluster = np.ones(len(filters[0][0]))*lambdas[j]
        for i in range(len(filters)):
            res_cluster*=coefs[j][i].dot(np.array(filters[i]))
        res+=res_cluster
    res = np.sum(np.log(res))
    return res


def pad_coefs(coefs):
    """
    Convertit coefs (liste irrégulière) en tableau NumPy (K, D, V_max)
    en paddant avec des zéros les valeurs inexistantes.
    À appeler une seule fois avant les appels répétés à q_z_x.
    """
    K   = len(coefs)
    D   = len(coefs[0])
    V_max = max(len(coefs[j][i]) for j in range(K) for i in range(D))

    padded = np.zeros((K, D, V_max))
    for j in range(K):
        for i in range(D):
            v = len(coefs[j][i])
            padded[j, i, :v] = coefs[j][i]
    return padded


def sampling_LCA(lambdas, coefs_padded, n_pop):
    """
    lambdas      : array-like (K,)
    coefs_padded : np.ndarray (K, D, V_max) — sortie de pad_coefs()
    n_pop        : int
    """
    K, D, V_max = coefs_padded.shape

    # Tirage des clusters — (N,)
    clusters = np.random.choice(K, n_pop, p=lambdas)

    # CDF des coefs pour chaque (k, d) — (K, D, V_max)
    cdf = np.cumsum(coefs_padded, axis=2)

    # Tirage uniforme — (N, D)
    u = np.random.uniform(size=(n_pop, D))

    # Sélection des CDF selon le cluster de chaque individu — (N, D, V_max)
    cdf_n = cdf[clusters]           # (N, D, V_max)

    # Inverse CDF : premier index où u < cdf
    # u[:, :, None] < cdf_n  →  (N, D, V_max) booléen
    return ((u[:, :, None] < cdf_n).argmax(axis=2)).T   # (N, D)

def inverse_dict_data_df(data, dict_inv, columns):
    data_df = pd.DataFrame()
    for i in range(len(dict_inv)):
        data_df[columns[i]]= (np.vectorize(dict_inv[i].get)(data[i]))
    return data_df

def train_sample_LCA(args):

    print("\n\nTraining and Sampling Latent Class Analysis (EM) on raw data\n\n")
    
    ##################
    ### Parameters ###
    ##################

    term = "_LCA"
    
    t0 = time()
    
    n_sample = args.n_generation
    
    filename_training = args.filename_training
    
    data_dir = get_data_dir(args)
    
    folder_sampling = get_folder_sampling(args, term)
    sampling_file = get_file_path_sampling(args, term)
    
    if (not os.path.exists(folder_sampling)):
        os.makedirs(folder_sampling)


    #################
    ### Load Data ###
    #################

    info = get_info_file(args)[["Type", "Variable_name", "Bin_size"]]
    
    name_cat = info[info["Type"].isin(["binary","category","bool"])].reset_index()["Variable_name"]
    name_num = info[info["Type"].isin(["int","float"])].reset_index()["Variable_name"]
    
    columns = info["Variable_name"]
    
    df_data = import_data(os.path.join(data_dir, filename_training), columns, columns) # All variables are treated as categorical
    
    df_data, _ = preprocessing_cat_data_dataframe_sampling(df_data, args.transform.cat_min_count, name_cat)
    
    df_data, dict_translation_inverse = transform_num_into_quantiles(df_data, name_num)
    df_data = transform_num_into_bins(df_data, name_num, args.transform_statistical.bins_num)
    
    data_np, dict_inv = data_to_numpy(df_data) #data_np size (N,d)

    ######################
    ### Initiate Model ###
    ######################
    
    filters = generate_filters(data_np)
    
    n_clusters = find_best_numbers_clusters(filters, data_np)
    
    print("Best number of clusters is",n_clusters)
    
    coefs, ls, LLs = get_LCA_coefs(data_np, n_clusters)
    
    plt.figure()
    plt.plot(LLs)
    plt.savefig(f'{folder_sampling}/loglikelihood.png')
    plt.close()
    
    np_sample = sampling_from_coefs(coefs, ls, n_sample)
    df_sample = inverse_dict_data_df(np_sample, dict_inv, df_data.columns.to_numpy())

    ###################
    ### Sample Data ###
    ###################    
    
    save_time(t0, args, term)
    
    df_sample = bins_to_values(df_sample, df_data, dict_translation_inverse, name_num)

    df_sample.to_csv(sampling_file,sep=";", index=False)