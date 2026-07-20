import pandas as pd
import numpy as np

from utils.utils_sample import preprocessing_cat_data_dataframe_sampling

from utils.utils_dir import get_file_path_sampling, get_data_dir
from utils.utils_import import import_data, get_info_file

def train_sample_Direct_Inflating(args):
    '''
    Replicate the training population at the desired size
    '''

    print("\nSampling by Direct Replication of the training data\n\n")
    
    ##################
    ### Parameters ###
    ##################

    term = "_Direct_Inflating"

    filename_training = args.filename_training

    n_sample = args.n_generation

    data_dir = get_data_dir(args)
        
    sampling_file = get_file_path_sampling(args, term)
    
    #################
    ### Load Data ###
    #################

    info = get_info_file(args)[["Type", "Variable_name", "Bin_size"]]
    
    name_cat = info[info["Type"].isin(["binary","category","bool"])].reset_index()["Variable_name"]
    
    columns = info["Variable_name"]
    df_data = import_data(os.path.join(data_dir,filename_training), columns, columns) # All variables are treated as categorical
    
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