import numpy as np
import pandas as pd
from tqdm import tqdm

def get_rate_of_impossible_combinations(df_generated, df_info, prop_ori, prop_gen, columns, combs, values):
    """
    Calculates the rate of impossible combinations in a generated dataset.

    Impossible combinations are cases where a combination of categorical values appears in 
    the generated dataset but was absent in the original dataset.

    Parameters:
    -----------
    df_generated : pd.DataFrame
        The generated dataset containing synthetic data.
    
    df_info : pd.DataFrame
        A DataFrame containing metadata or additional information about the dataset.
        Specifically, it includes geographic attributes under the "Geographical_attribute" column.
    
    prop_ori : np.array
        An array representing the proportions of each combination in the original dataset.
        Used to identify combinations that do not exist in the original data (i.e., prop_ori == 0).
    
    prop_gen : np.array
        An array representing the proportions of each combination in the generated dataset.
        Helps identify combinations that were generated but do not exist in the original dataset 
        (i.e., prop_ori == 0 and prop_gen != 0).
    
    columns : pd.Series or np.array
        A list or array of column names from df_generated, representing different categorical features.
        Used to identify which columns form the impossible combinations.
    
    combs : np.array
        An array representing all possible unique combinations of categorical variables in the dataset.
        The function filters out combinations that exist in df_generated but were absent in the original data.
    
    values : np.array
        An array containing the actual values corresponding to the combs array.
        It holds the specific categorical values that form impossible combinations.

    Returns:
    --------
    tuple (float, float)
        - Rate of impossible combinations **without** geographic attributes.
        - Rate of impossible combinations **with** geographic attributes.
    """
    df_final = df_generated.copy().astype(str)
    df_final["idx"] = df_final.index 
    print(len(prop_ori), len(prop_gen), len(combs))
    prob_combs = combs[(prop_ori==0)&(prop_gen!=0)]
    prob_values = values[(prop_ori==0)&(prop_gen!=0)].astype(str)

    columns_list = columns.to_numpy()
    geo_array = df_info["Geographical_attribute"].to_numpy()
    prob_cols = columns_list[prob_combs.astype(int)]
    prob_combs_unique = np.unique(prob_combs.astype(int), axis = 0)
    has_geo = geo_array[prob_combs_unique.astype(int)].any(1)
    prob_combs_non_geo_unique = prob_combs_unique[~has_geo]
    prob_combs_with_geo_unique = prob_combs_unique[has_geo]
    prob_cols_with_geo_unique = columns_list[prob_combs_with_geo_unique]
    prob_cols_non_geo_unique = columns_list[prob_combs_non_geo_unique]
    for i in tqdm(range(len(prob_cols_non_geo_unique))):
        cols = prob_cols_non_geo_unique[i]
        prob_values_comb = prob_values[(prob_cols == cols).all(1)]
        bool_array = df_final.index.isin((df_final).merge(pd.DataFrame(prob_values_comb, columns=cols), how='inner').idx)
        df_final = df_final[~bool_array]
    df_final_non_geo = df_final.copy()
    for i in tqdm(range(len(prob_cols_with_geo_unique))):
        cols = prob_cols_with_geo_unique[i]
        prob_values_comb = prob_values[(prob_cols == cols).all(1)]
        bool_array = df_final.index.isin((df_final).merge(pd.DataFrame(prob_values_comb, columns=cols), how='inner').idx)
        df_final = df_final[~bool_array]
    return 1-len(df_final_non_geo)/len(df_generated), 1-len(df_final)/len(df_generated)

def get_proportion_from_original_data(original_data, generated_data):
    '''
    Goal:
    Proportion of generated samples that are identical to a sample from the original data

    Input:
    original_data: either the full dataset or the training dataset
    The training dataset allows to compute how original the samples are
    The full original dataset allows to compute to what extend, the samples are close to the reality/realistic

    generated_data: the synthetic population generated
    
    Output:
    Proportion of generated samples that are identical to a sample from the original data
    '''
    res = np.zeros(len(generated_data),bool)
    for i in tqdm(range(len(generated_data))):
        res[i] = (generated_data[i]==original_data).all(1).any()
    return np.mean(res)

def get_proportion_from_original_data_df(df_original_data, df_generated_data, columns):
    '''
    Goal:
    Proportion of generated samples that are identical to a sample from the original data

    Input:
    original_data: either the full dataset or the training dataset
    The training dataset allows to compute how original the samples are
    The full original dataset allows to compute to what extend, the samples are close to the reality/realistic

    generated_data: the synthetic population generated
    
    Output:
    Proportion of generated samples that are identical to a sample from the original data
    '''
    df_generated_data = df_generated_data[columns]
    df_generated_data = df_generated_data.value_counts().reset_index()
    count = df_generated_data["count"]
    df_generated_data = df_generated_data[columns]
    df_original_data = df_original_data[columns].drop_duplicates()
    count_tot = pd.Series(0, index=df_original_data.index)
    data_concat = pd.concat([df_original_data,df_generated_data])
    count_concat = pd.concat([count_tot,count])
    n_copy = count_concat[data_concat.duplicated()].sum()
    n = count_concat.sum()
    return n_copy/n


def get_proportion_from_original_data_df_not_in_other_df_previous(df_original_data, df_execpt_data, df_generated_data, columns):
    '''
    Goal:
    Proportion of generated samples that are identical to a sample from the original data

    Input:
    original_data: either the full dataset or the training dataset
    The training dataset allows to compute how original the samples are
    The full original dataset allows to compute to what extend, the samples are close to the reality/realistic

    generated_data: the synthetic population generated
    
    Output:
    Proportion of generated samples that are identical to a sample from the original data 
    # \sum_{\hat x\in \hat X}  1_{\hat x \in X\ X_{train]}/|\hat X|
    '''
    df_generated_data = df_generated_data[columns]
    df_generated_data = df_generated_data.value_counts().reset_index()
    count = df_generated_data["count"] # \hat X
    df_generated_data = df_generated_data[columns]
    df_original_data = df_original_data[columns].drop_duplicates()
    df_execpt_data = df_execpt_data[columns].drop_duplicates()
    count_tot = pd.Series(0, index=df_original_data.index) # X
    count_tot_execpt = pd.Series(0, index=df_execpt_data.index) # X train
    data_concat_0 = pd.concat([df_original_data,df_generated_data])
    count_concat_0 = pd.concat([count_tot,count])
    data_concat_1 = pd.concat([df_execpt_data,data_concat_0[data_concat_0.duplicated()]])
    count_concat_1 = pd.concat([count_tot_execpt,count_concat_0[data_concat_0.duplicated()]])
    n_copy = count_concat_0[data_concat_0.duplicated()].sum() # \sum_{\hatx\in\hat X} 1_{\hat x \in X}
    n_copy_both = count_concat_1[data_concat_1.duplicated()].sum() # \sum_{\hatx\in\hat X} 1_{\hat x \in X \cup X_{train]}}
    n = count_concat_0.sum() # |\hat X|
    return (n_copy-n_copy_both)/n # \sum_{\hatx\in\hat X} 1_{\hat x \in X} -  \sum_{\hatx\in\hat X} 1_{\hat x \in X \cup X_{train]}}/|\hat X|
                                #  = \sum_{\hat x\in \hat X}  1_{\hat x \in X\ X_{train]}/|\hat X|

def get_proportion_from_original_data_df_not_in_other_df(df_original_data, df_execpt_data, df_generated_data, columns):
    '''
    Goal:
    Proportion of generated samples that are identical to a sample from the original data

    Input:
    original_data: either the full dataset or the training dataset
    The training dataset allows to compute how original the samples are
    The full original dataset allows to compute to what extend, the samples are close to the reality/realistic

    generated_data: the synthetic population generated
    
    Output:
    Proportion of generated samples that are identical to a sample from the original data
    '''
    df_generated_data = df_generated_data[columns]
    df_generated_data = df_generated_data.value_counts().reset_index()
    count = df_generated_data["count"] # \hat X
    df_generated_data = df_generated_data[columns]
    df_original_data = df_original_data[columns].drop_duplicates()
    df_execpt_data = df_execpt_data[columns].drop_duplicates()
    count_tot = pd.Series(0, index=df_original_data.index) # X
    count_tot_execpt = pd.Series(0, index=df_execpt_data.index) # X train
    data_concat_0 = pd.concat([df_original_data,df_generated_data])
    count_concat_0 = pd.concat([count_tot,count])
    data_concat_1 = pd.concat([df_execpt_data,data_concat_0[data_concat_0.duplicated()]])
    count_concat_1 = pd.concat([count_tot_execpt,count_concat_0[data_concat_0.duplicated()]])
    data_concat_2 = pd.concat([df_execpt_data,df_generated_data])
    count_concat_2 = pd.concat([count_tot_execpt,count])
    n_copy = count_concat_0[data_concat_0.duplicated()].sum() # \sum_{\hatx\in\hat X} 1_{\hat x \in X}
    n_copy_both = count_concat_1[data_concat_1.duplicated()].sum() # \sum_{\hatx\in\hat X} 1_{\hat x \in X \cup X_{train]}}
    n_copy_in_generated = count_concat_2[data_concat_2.duplicated()].sum() # \sum_{\hatx\in\hat X} 1_{\hat x \in X_{train]}}
    n = count.sum() # |\hat X|
    print(n, n_copy, n_copy_both, n_copy_in_generated)
    if (n==n_copy_in_generated):
        return "NA"
    else:
        return (n_copy-n_copy_both)/(n-n_copy_in_generated) 
                                # \sum_{\hatx\in\hat X} 1_{\hat x \in X} -  \sum_{\hatx\in\hat X} 1_{\hat x \in X \cup X_{train]}}/(|\hat X|-\sum_{\hatx\in\hat X} 1_{\hat x \in X_{train]}})
                                #  = \sum_{\hat x\in \hat X}  1_{\hat x \in X\ X_{train]}/(\sum_{\hat x\in\hat X} 1_{\hat x \in \hatX\X_{train]}})

def get_0_proportion_respected(prop_original_data, prop_generated_data):
    '''
    Goal:
    For a given size of comparison (marginal, bivariate, trivariate) computes the proportion of combination in the generated data that is not present in the original data, ie with 0% of apparition. 

    Input:
    prop_original_data: either the full dataset or the training dataset
    The training dataset allows to compute how original the samples are from the training data
    The full original dataset allows to compute to what extend the model has guesses some new combinations

    prop_generated_data: proportion of the synthetic population generated
    
    Output:
    Number of combinations in the generated population that are not present in the original data, number of combinations that are not present in the original data, number of combinations present in the sampled data
    '''
    n_zeros_original = np.sum(prop_original_data==0)
    n_non_zeros_samples_zeros_original = np.sum((prop_generated_data>0)&(prop_original_data==0))
    n_non_zeros_samples = np.sum((prop_generated_data>0))
    return n_non_zeros_samples_zeros_original,n_zeros_original, n_non_zeros_samples
