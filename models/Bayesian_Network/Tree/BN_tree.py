import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
import os

from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import TreeSearch

from utils.utils_train import transform_num_into_bins, transform_num_into_quantiles
from utils.utils_sample import preprocessing_cat_data_dataframe_sampling, bins_to_values


def train_sample_BN_tree(args):

    print("\n\nTraining and Sampling Bayesian Network with Tree structure research on raw data\n\n")
        
    ##################
    ### Parameters ###
    ##################
    
    term = "_Bayesian_Network_Tree"
    datapath = "Data"
    dataname = args.dataname
    filename = args.filename
    infoname = args.infoname
    attr_setname = args.attributes_setname
    n_sample = args.n_generation

    info_path = f'{datapath}/{dataname}/{infoname}'
    
    dataset_path = f'{datapath}/{dataname}/{filename}'
        
    filename_sampling = (args.sampling_terminaison+"_"+str(n_sample)+term+".").join(args.filename.split('.'))

    folder_sampling = f'{args.sample_folder}/{args.folder_save+term}'
    sampling_file = f'{folder_sampling}/{filename_sampling}'
    
    if (not os.path.exists(folder_sampling)):
        os.makedirs(folder_sampling)

    #################
    ### Load Data ###
    #################
    
    info = pd.read_csv(info_path, sep = ";")
    info = info[info[attr_setname]][["Type", "Variable_name", "Bin_size"]]
    
    name_cat = info[info["Type"].isin(["binary","cat","bool"])].reset_index()["Variable_name"]
    name_num = info[info["Type"].isin(["cont","int","float"])].reset_index()["Variable_name"]
    
    df_data = pd.read_csv(dataset_path, sep=";", low_memory=False)[info["Variable_name"]].astype(str)
    
    df_data, _ = preprocessing_cat_data_dataframe_sampling(df_data, args.transform.cat_min_count, name_cat)
    
    df_data, dict_translation_inverse = transform_num_into_quantiles(df_data, name_num)
    df_data = transform_num_into_bins(df_data, name_num, args.transform_statistical.bins_num)

    ######################
    ### Initiate Model ###
    ######################

    tree = TreeSearch(df_data,)

    #######################
    ### Learn Structure ###
    #######################

    best_model = tree.estimate()

    nx.draw_circular(
        best_model, with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight="bold"
    )
    plt.savefig(f"{folder_sampling}/best_graph.png")
    plt.close()
    
    ########################
    ### Learn Transition ###
    ########################

    model = BayesianNetwork(best_model)
    model.fit(df_data)

    ###################
    ### Sample Data ###
    ###################    
    
    df_sample = (BayesianModelSampling(model).forward_sample(n_sample))

    df_sample = bins_to_values(df_sample, df_data, dict_translation_inverse, name_num)    
    
    df_sample.to_csv(sampling_file,sep=";", index=False)