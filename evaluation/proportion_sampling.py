from evaluation.metrics_proportion import SRMSE, Pearson, R2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def get_combinations(a,b,n_combi):
    '''
    a,b: 2 tabular data of dimension (#a,n_attributes) and (#b, n_attributes)
    n_combi: number of values in the combinations
    '''
    _,dim = a.shape
    if n_combi==0:
        return np.array([[]]), np.array([[]])
    else:
        combi_idx, combi_values = [], []
        for i in range(dim-n_combi+1):
            arr_unique = np.unique(np.concatenate([a[:,i],b[:,i]]))
            combi_idx_n, combi_values_n = get_combinations(a[:,i+1:],b[:,i+1:],n_combi-1)
            for combi_idx_1,combi_value_1 in zip(combi_idx_n,combi_values_n):
                for val in arr_unique:
                    new_comb_idx_1 = np.concatenate([np.array([i]),1+i+combi_idx_1])
                    new_comb_value_1 = np.concatenate([np.array([val]),combi_value_1])
                    combi_idx.append(new_comb_idx_1)
                    combi_values.append(new_comb_value_1)
        return np.array(combi_idx),np.array(combi_values)

def get_combinations_col(dim_list,n_combi):
    '''
    dim_list: dimension of data / # of variables
    n_combi: number of values in the combinations
    '''
    if n_combi==1:
        return [[i] for i in range(dim_list)]
    else:
        list_n_minus_1 = get_combinations_col(dim_list-1,n_combi-1)
        combi_idx = []
        for combi_idx_n_ind in tqdm(list_n_minus_1):
            for i in range(combi_idx_n_ind[-1]+1,dim_list):
                combi_idx_n_ind.append(i)
                combi_idx.append(combi_idx_n_ind.copy())
                combi_idx_n_ind.pop()
        return combi_idx
   

def generate_dict_from_unique_list(unique_list):
    res = {}
    for i,a in enumerate(unique_list): 
        res[a] = i
    return res



def scores_for_x_joint_attributes(preprocessed_original_data,preprocessed_generated_data, n_attributes, folder):
    '''
    Input: 
    preprocessed_original_data: data from the test set that are already been preprocessed into bins for numerical values
    preprocessed_generated_data: generated data that are already been preprocessed into bins for numerical values with the same bins as the test data
    n_attributes: number of attributes the joint distribution

    Output:
    SRMSE, Pearson, R2 for n_attributes
    '''
    n_original, n_dim = preprocessed_original_data.shape
    n_generated, n_dim = preprocessed_generated_data.shape
    unique_arr = []
    for i in tqdm(range(n_dim)):
        x = np.concatenate([preprocessed_original_data[:,i], preprocessed_generated_data[:,i]])
        x = np.unique(x)
        x = x[~pd.isna(x)]
        unique_arr.append(x)
    print("Generation possible combi...")
    combi_idx = get_combinations_col(n_dim, n_attributes)
    print("Generation possible combi over")
    print("Generation dictionaries...")
    dict_list = []
    for a in tqdm(unique_arr):
        dict_arr = {}
        for i,b in enumerate(a):
            dict_arr[b] = i
        dict_list.append(dict_arr)
    print("Generation dictionaries over")
    print("Convertion into num...")
    process_original_data = np.zeros_like(preprocessed_original_data,int)
    process_generated_data = np.zeros_like(preprocessed_generated_data,int)
    for i in tqdm(range(n_dim)):
        process_original_data[:,i] = np.vectorize(dict_list[i].get)(preprocessed_original_data[:,i])
        process_generated_data[:,i] = np.vectorize(dict_list[i].get)(preprocessed_generated_data[:,i])
    print("Convertion into num over")
    print("Generation proportion...")
    len_unique_arr = np.array([len(a) for a in unique_arr])
    prop_original = np.array([])
    prop_generated = np.array([])
    combi = []
    value = []
    idx = np.zeros(n_attributes,int)
    for comb in tqdm(combi_idx):
        prop_original_temp = np.zeros(len_unique_arr[comb])
        prop_generated_temp = np.zeros(len_unique_arr[comb])
        for a_val in (process_original_data):
            idx = a_val[comb]
            prop_original_temp[tuple(idx)]+=1
        prop_original = np.concatenate([prop_original,prop_original_temp.reshape(-1)/n_original])
        for a_val in (process_generated_data):
            idx = a_val[comb]
            prop_generated_temp[tuple(idx)]+=1
        prop_generated = np.concatenate([prop_generated,prop_generated_temp.reshape(-1)/n_generated])
        shape = list(prop_generated_temp.reshape(-1).shape)
        shape.append(len(comb))
        combi_temp = np.zeros(shape)
        combi_temp[:] = comb
        if len(combi)>0:
            combi = np.concatenate([combi,combi_temp])
        else:
            combi = combi_temp.copy()
        shape = list(prop_generated_temp.shape)
        shape.append(len(comb))
        value_temp = np.zeros(shape, str)
        for i,id in enumerate(comb):
            for j,val in enumerate(unique_arr[id]):
                idx = [slice(None)]*value_temp.ndim
                idx[i] = j
                idx[-1] = i
                value_temp[tuple(idx)] = val
        if len(value)>0:
            value = np.concatenate([value, value_temp.reshape(-1,len(comb))])
        else:
            value = value_temp.reshape(-1,len(comb)).copy()
    prop_original_data = np.array(prop_original)
    prop_generated_data = np.array(prop_generated)
    print("Generation proportion over")
    print("Generation score...")
    SRMSE_score, Pearson_score,R2_score = SRMSE(prop_original_data, prop_generated_data, len(prop_original_data)), Pearson(prop_original_data,prop_generated_data), R2(prop_original_data,prop_generated_data)
    print("Genration score over")

    dir_path = f'../Results/{folder}'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    np.save(f"{dir_path}/prop_original_data.npy",prop_original_data)
    np.save(f"{dir_path}/prop_generated_data.npy",prop_generated_data)
    np.save(f"{dir_path}/comb.npy",combi)
    np.save(f"{dir_path}/values.npy",value)

    np.savetxt(f"{dir_path}/scores.txt",[SRMSE_score, Pearson_score, R2_score])

    plt.figure(figsize=[7,7])
    plt.plot([-1,1], [-1,1], color='r', alpha=0.5)
    plt.scatter(prop_original_data,prop_generated_data)
    plt.xlim(-0.02,np.max([prop_generated_data, prop_original_data])+0.02)
    plt.ylim(-0.02,np.max([prop_generated_data, prop_original_data])+0.02)
    plt.xlabel("Original data proportion")
    plt.ylabel("Sampled data proportion")
    plt.savefig(f"{dir_path}/comparison_prop.png")
    plt.close()
    return SRMSE_score, Pearson_score, R2_score


    
def get_combinations_from_ordered_unique_list(unique_list,n_combi):
    '''
    a,b: 2 tabular data of dimension (#a,n_attributes) and (#b, n_attributes)
    n_combi: number of values in the combinations
    '''
    if n_combi==1:
        combi_idx, combi_values = [], []
        print("One element generation")
        for i,a in tqdm(enumerate(unique_list), total=len(unique_list)):
            for b in a:
                combi_idx.append([i])
                combi_values.append([b])
        return combi_idx,combi_values
    else:
        combi_idx_n, combi_values_n = get_combinations_from_ordered_unique_list(unique_list[:-1],n_combi-1)
        combi_idx, combi_values = [], []
        print(f"{n_combi} element generation")
        for combi_idx_n_ind, combi_values_n_ind in tqdm(zip(combi_idx_n, combi_values_n), total = len(combi_idx_n)):
            for i,a in enumerate(unique_list[combi_idx_n_ind[-1]+1:]):
                for b in a:
                    combi_idx_n_ind.append(i)
                    combi_values_n_ind.append(b)
                    combi_idx.append(combi_idx_n_ind.copy())
                    combi_values.append(combi_values_n_ind.copy())
                    combi_idx_n_ind.pop()
                    combi_values_n_ind.pop()
        return combi_idx,combi_values





def generate_proportion_from_dataset(preprocessed_original_full_data, target_dataset, n_attributes, folder, name, save_value_combi = False):
    '''
    Input: 
    preprocessed_original_data: all data that are already been preprocessed into bins for numerical values. 
    The score is not computed, but the original data is used to find all possible values

    target_dataset: generated data that are already been preprocessed into bins for numerical values with the same bins as the test data
    
    n_attributes: number of attributes the joint distribution

    folder: folder to save the results

    name: name of the output file

    save_value_combi (optionnal): save the combinations and the values for all combinations of columns in the same order than the proportion generated.
    Useful to verify the bins or to investigate where an error could come from

    Output:
    write in a file the proportion for the target_dataset in folder/name.npy
    '''
    _, n_dim = preprocessed_original_full_data.shape
    n_target, n_dim = target_dataset.shape
    unique_arr = []
    for i in tqdm(range(n_dim)):
        x = preprocessed_original_full_data[:,i]
        x = np.unique(x)
        x = x[~pd.isna(x)]
        unique_arr.append(x)
    print("Generation possible combi...")
    combi_idx = get_combinations_col(n_dim, n_attributes)
    print("Generation possible combi over")
    print("Generation dictionaries...")
    dict_list = []
    for a in tqdm(unique_arr):
        dict_arr = {}
        for i,b in enumerate(a):
            dict_arr[b] = i
        dict_list.append(dict_arr)
    print("Generation dictionaries over")
    print("Convertion into num...")
    process_target_data = np.zeros_like(target_dataset,int)
    for i in tqdm(range(n_dim)):
        try:
            process_target_data[:,i] = np.vectorize(dict_list[i].get)(target_dataset[:,i])
        except:
            dict_conv = {}
            target_values = np.unique(preprocessed_original_full_data[:,i])
            for val in np.unique(target_dataset[:,i]):
                target = min(np.max(target_values),val)
                target = np.min(target_values[target_values>=target])
                dict_conv[val] = target
            target_dataset[:,i] = np.vectorize(dict_conv.get)(target_dataset[:,i])
            process_target_data[:,i] = np.vectorize(dict_list[i].get)(target_dataset[:,i])
    print("Convertion into num over")
    print("Generation proportion...")
    len_unique_arr = np.array([len(a) for a in unique_arr])
    prop_target = np.array([])
    combi = []
    value = []
    idx = np.zeros(n_attributes,int)
    for comb in tqdm(combi_idx):
        prop_target_temp = np.zeros(len_unique_arr[comb])
        for val in (process_target_data):
            idx = val[comb]
            prop_target_temp[tuple(idx)]+=1
        prop_target = np.concatenate([prop_target,prop_target_temp.reshape(-1)/n_target])
        if (save_value_combi):
            shape = list(prop_target_temp.reshape(-1).shape)
            shape.append(len(comb))
            combi_temp = np.zeros(shape)
            combi_temp[:] = comb
            if len(combi)>0:
                combi = np.concatenate([combi,combi_temp])
            else:
                combi = combi_temp.copy()
            shape = list(prop_target_temp.shape)
            shape.append(len(comb))
            value_temp = np.zeros(shape, object)
            for i,id in enumerate(comb):
                for j,val in enumerate(unique_arr[id]):
                    idx = [slice(None)]*value_temp.ndim
                    idx[i] = j
                    idx[-1] = i
                    value_temp[tuple(idx)] = val
            if len(value)>0:
                value = np.concatenate([value, value_temp.reshape(-1,len(comb))])
            else:
                value = value_temp.reshape(-1,len(comb)).copy()
    print("Generation proportion over")

    dir_path = folder

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    prop_generated_data = np.array(prop_target)
    print(f"Save path: {dir_path}/{name}_{n_attributes}.npy")
    np.save(f"{dir_path}/{name}_{n_attributes}.npy",prop_generated_data)
    if (save_value_combi):
        np.save(f"{dir_path}/{name}_{n_attributes}_comb.npy",combi)
        np.save(f"{dir_path}/{name}_{n_attributes}_values.npy",value)   