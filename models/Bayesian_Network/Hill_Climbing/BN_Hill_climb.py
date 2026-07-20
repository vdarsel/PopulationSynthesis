import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
import os
from time import time

from pgmpy.models import BayesianNetwork

from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import  HillClimbSearch, BicScore

from utils.utils_train import transform_num_into_bins, transform_num_into_quantiles
from utils.utils_sample import preprocessing_cat_data_dataframe_sampling, bins_to_values


from utils.utils_dir import get_data_dir, get_folder_sampling, get_file_path_sampling
from utils.utils_import import get_info_file, import_data

from utils.utils_time import save_time


def train_sample_BN_hill_climb(args):

    print("\n\nTraining and Sampling Bayesian Network with Hill Climbing structure research on raw data\n\n")
    
    ##################
    ### Parameters ###
    ##################

    term = "_Bayesian_Network_Hill_Climb"

    t0 = time()
    
    n_sample = args.n_generation
    
    filename_training = args.filename_training
    
    data_dir = get_data_dir(args)

    folder_sampling = get_folder_sampling(args, term)
    sampling_file = get_file_path_sampling(args, term)
    
    #################
    ### Load Data ###
    #################
    
    info = get_info_file(args)[["Type", "Variable_name", "Bin_size"]]
    
    name_cat = info[info["Type"].isin(["binary","category","boolean"])].reset_index()["Variable_name"]
    name_num = info[info["Type"].isin(["int","float"])].reset_index()["Variable_name"]

    columns = info["Variable_name"]
    
    df_data = import_data(os.path.join(data_dir,filename_training), columns, columns) # All variables are treated as categorical
    
    df_data, _ = preprocessing_cat_data_dataframe_sampling(df_data, args.transform.cat_min_count, name_cat)
    
    df_data, dict_translation_inverse = transform_num_into_quantiles(df_data, name_num)
    df_data = transform_num_into_bins(df_data, name_num, args.transform_statistical.bins_num)

    ######################
    ### Initiate Model ###
    ######################

    hc = HillClimbSearch(df_data)

    #######################
    ### Learn Structure ###
    #######################
    
    best_model = hc.estimate(scoring_method=BicScore(df_data),)

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