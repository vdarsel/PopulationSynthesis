import sklearn.preprocessing
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline

CAT_RARE_VALUE = '__rare__'

def process_nans(X_training: np.ndarray, col_num: np.ndarray, col_cat: np.ndarray, policy_num: str, policy_cat: str) -> np.ndarray:

    assert policy_num is not None
    if policy_num == 'drop-rows':
        X_training = X_training[~pd.isna(X_training[:,col_num]).any(1)]
    elif policy_num == 'mean':
        # print(X_training,col_num)
        # print(X_training[:,col_num].astype(float))
        # X_training[:,col_num] = X_training[:,col_num].astype(float)
        new_values = np.nanmean(X_training[:,col_num], axis=0)
        # print(new_values)
        for i,val in enumerate(new_values):
            X_training[:,col_num[i]] = np.where(pd.isna(X_training[:,col_num[i]]),val,X_training[:,col_num[i]])
    else:
        raise(f'Unknown policy for num nan {policy_num}')
    assert policy_cat is not None
    if policy_cat == 'drop-rows':
        X_training = X_training[~pd.isna(X_training[:,col_cat]).any(1)]
    elif policy_cat =="most_frequent":
        for col_i in col_cat:
            X_training_not_nan = X_training[:,col_i][~pd.isna(X_training[:,col_i])]
            uni_val, counts = np.unique(X_training_not_nan, return_counts=True)
            val = uni_val[np.argmax(counts)]
            X_training[:,col_i] = np.where(pd.isna(X_training[:,col_i]),val,X_training[:,col_i])
    else:
        raise(f'Unknown policy for cat nan {col_cat}')
    return X_training



def get_normalizer_num(
    X_training: np.ndarray, normalization: str, seed: int = 0):
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'minmax':
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X_training.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    elif normalization == 'normal_transform':
        normalizer = sklearn.preprocessing.QuantileTransformer_bis(
            output_distribution='normal',
            n_quantiles=max(min(X_training.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise(f'Unknown normalization {normalization}')

    normalizer.fit(X_training)
    return normalizer.inverse_transform

def get_categories_inverse(X_training: np.ndarray):
    unknown_value = np.iinfo('int64').max - 3
    oe = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',  # type: ignore[code]
        unknown_value=unknown_value,  # type: ignore[code]
        dtype='int64',  # type: ignore[code]
    ).fit(X_training)
    encoder = make_pipeline(oe)
    encoder.fit(X_training)
    return encoder.inverse_transform

def preprocessing_cat_data_dataframe_sampling(dataset_ref: pd.DataFrame, min_count, cols_cat, other_dataset : list[pd.DataFrame] = []):
    if(len(cols_cat)>0):
        X_new = dataset_ref.copy()
        X_new_list = [df.copy() for df in other_dataset]
        print("Initial categories (training):", dataset_ref[cols_cat].nunique().to_list())
        if (min_count is not None):
            for column_idx in (cols_cat):
                value_counts = dataset_ref[column_idx].value_counts()
                popular_categories = value_counts[value_counts>=min_count].index
                X_new.loc[~X_new[column_idx].isin(popular_categories),column_idx] = CAT_RARE_VALUE
                for df in X_new_list:
                    df.loc[~df[column_idx].isin(popular_categories),column_idx] = CAT_RARE_VALUE
        print("Final categories (training):", X_new[cols_cat].nunique().to_list())
    return X_new, X_new_list

def bins_to_values(data_generated: pd.DataFrame, data_train: pd.DataFrame, inverse_dict_list: list[dict], name_num: list):
    for inverse_dict,name in zip(inverse_dict_list, name_num):
        unique_val_train = data_train[name].unique()
        bin_values = np.sort(unique_val_train)
        bin_values_0 = bin_values
        bin_values_1 = np.concatenate([bin_values,[1]])[1:]
        for b_0,b_1 in zip(bin_values_0, bin_values_1):
            idx_bin = data_generated[name]==b_0
            n_bin = np.sum(idx_bin)
            data_generated.loc[idx_bin,name] = np.random.rand(n_bin)*(b_1-b_0)+b_0
        values = np.array([a for a in inverse_dict.keys()])
        res = []
        for val in data_generated[name]:
            res.append(inverse_dict[np.max(values[values<val])])
        data_generated[name] = res
    return data_generated