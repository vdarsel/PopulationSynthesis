import numpy as np
import pandas as pd
import os
from itertools import combinations
from tqdm import tqdm
from math import comb

def compute_proportion_file_from_unique_array_and_df(
    dict_unique_values,
    preprocessed_data_df,
    columns,
    name,
    n_attributes,
    folder,
    save_combi=False):
    '''dict_unique_values: dictionary with unique values for all columns
    preprocessed_generated_data_df: data to compute the frequencies (dataframe)
    name: name of the saving file
    n_attributes: number of attributes for joint distribution
    folder: folder to save frequencies
    
    Compute and save the proportions from the dictionaries
    '''
    print(f"Generation of the proportions ({n_attributes} attribute(s))...")
    combis = (combinations(np.arange(len(columns)),n_attributes))
    values = []
    freq_list = []
    combis_list = []
    
    pbar = tqdm(combis, total=comb(len(columns), n_attributes))
    pbar.set_description("Computing proportions")
    
    for combination in pbar:
        cols = columns[np.array(combination)]
        freq_serie = pd.Series(0.0, index = pd.MultiIndex.from_product(
            [dict_unique_values[col] for col in cols]
        ))
        freq_temp = preprocessed_data_df[cols].value_counts(normalize=True)
        freq_serie.loc[freq_temp.index] = freq_temp
        freq_values = freq_serie.to_numpy()
        
        freq_list.append(freq_values)
        values.append([np.array(a) for a in (freq_serie.index)])
        combis_list.append([np.array(combination) for _ in (freq_serie.index)])
        
    freq_list = np.concat(freq_list)
    
    dir_path = folder

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    np.save(f"{dir_path}/{name}_{n_attributes}.npy",freq_list)

    if save_combi:
        np.save(f"{dir_path}/{name}_{n_attributes}_comb.npy",np.concatenate(combis_list))
        np.save(f"{dir_path}/{name}_{n_attributes}_values.npy",np.concatenate(values), allow_pickle=True)
        
        
    print("Generation of the proportions done")


def recover_lists_from_dictionnary(columns, dicts_unique, concatenate_values, n_attributes):
    combis = (combinations(np.arange(len(columns)),n_attributes))
    result = []
    n_index_0 = 0
    for combination in combis:
        cols = columns[np.array(combination)]
        n_index_plus = np.prod([len(dicts_unique[col]) for col in cols])
        n_index_1 = n_index_0 + n_index_plus
        result.append(concatenate_values[n_index_0: n_index_1])
        n_index_0 = n_index_1
    return result
