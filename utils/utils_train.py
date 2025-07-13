import numpy as np
import os

import src
from torch.utils.data import Dataset

import pandas as pd
import torch


def preprocessing_cat_data(X_cat: dict[np.ndarray], X_num: dict[np.ndarray], min_size, idx, method):
    if(len(idx)>0):
        print("Initial length (training/validation):", len(X_cat['train']), len(X_cat['validation']))
        print(X_cat['train'].shape, X_cat['validation'].shape)
        print(X_num['train'].shape, X_num['validation'].shape)
        for id in idx:
            existing_values = np.unique(np.concatenate([X_cat['train'][:,id], X_cat['validation'][:,id]]))
            for value in existing_values:
                bool_array = (X_cat['train'][:,id] == value)
                n_val = np.sum(bool_array)
                if n_val<min_size:
                    if method=="cut":
                        X_cat['train'] = X_cat['train'][~bool_array]
                        X_num['train'] = X_num['train'][~bool_array]
                        X_num['validation'] = X_num['validation'][X_cat['validation'][:,id]!=value]
                        X_cat['validation'] = X_cat['validation'][X_cat['validation'][:,id]!=value]
                    else:
                        raise("Undefined method")
        print("Length after preprocessing (training/validation):", len(X_cat['train']), len(X_cat['validation']))
        print(X_cat['train'].shape, X_cat['validation'].shape)
        print(X_num['train'].shape, X_num['validation'].shape)
    return X_cat,X_num

def preprocessing_cat_data_df(X_cat: dict[pd.DataFrame], X_num: dict[pd.DataFrame], min_size, idx, method):
    if(len(idx)>0):
        print("Initial length (training/validation):", len(X_cat['train']), len(X_cat['validation']))
        for id in idx:
            existing_values = np.unique(np.concatenate([X_cat['train'][id].unique(), X_cat['validation'][id].unique()]))
            for value in existing_values:
                bool_serie = (X_cat['train'][id] == value)
                n_val = bool_serie.sum()
                if n_val<min_size:
                    if method=="cut":
                        X_cat['train'] = X_cat['train'][~bool_serie]
                        X_num['train'] = X_num['train'][~bool_serie]
                        X_num['validation'] = X_num['validation'][X_cat['validation'][id]!=value]
                        X_cat['validation'] = X_cat['validation'][X_cat['validation'][id]!=value]
                    else:
                        raise("Undefined method")
        print("Length after preprocessing (training/validation):", len(X_cat['train']), len(X_cat['validation']))
        # # print(X_cat['train'].shape, X_cat['validation'].shape)
        # # print(X_num['train'].shape, X_num['validation'].shape)
        # X_cat["train"].to_csv("ckpt/cat_data_after_preprocessing.csv", sep=";")
        # X_num["train"].to_csv("ckpt/num_data_after_preprocessing.csv", sep=";")
    return X_cat,X_num


        
class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]

def preprocess(dataset_path, basename, idx_cat, idx_num, T_dict, inverse = False, cat_encoding = None, get_transform = False):
    

    T = src.Transformations(**T_dict)

# def make_dataset(
#     data_path: str,
#     file_basename: str,
#     T: src.Transformations,
#     change_val: bool,
#     idx_cat: list,
#     idx_num: list
# ):
    dataset = make_dataset(
        data_path = dataset_path,
        file_basename= basename,
        T = T,
        change_val = False,
        idx_cat = idx_cat,
        idx_num = idx_num
    )
    

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat
        if X_num is None:
            X_num = {"train":np.zeros((len(X_cat["train"]),0)),"validation":np.zeros((len(X_cat["validation"]),0))}
        X_train_num, X_validation_num = X_num['train'], X_num['validation']
        X_train_cat, X_validation_cat = X_cat['train'], X_cat['validation']

        
        categories = src.get_categories(np.concatenate([X_train_cat,X_validation_cat]))
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_validation_num)
        X_cat = (X_train_cat, X_validation_cat)


        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            cat_inverse = dataset.cat_transform.inverse_transform

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse

        if get_transform:
            if (len(idx_num)>0):
                num_fctn = dataset.num_transform.transform
            else:
                num_fctn = lambda x: x
            cat_fctn = dataset.cat_transform.transform

            return X_num, X_cat, categories, d_numerical, num_fctn, cat_fctn
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)



def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    data_path: str,
    file_basename: str,
    T: src.Transformations,
    change_val: bool,
    idx_cat: list,
    idx_num: list,
):

    X_cat = {} if len(idx_cat)>0  else None
    X_num = {} if len(idx_num)>0 else None

    for split in ['train', 'validation']:
        filename = f'_{split}.'.join(file_basename.split('.'))
        # file = os.path.join(data_path,file_basename)
        
        X_num_t, X_cat_t = src.read_data(data_path, filename, idx_cat, idx_num)
        if X_num is not None:
            X_num[split] = X_num_t
        if X_cat is not None:
            X_cat[split] = X_cat_t  

    print((X_cat["train"]["educ_level"].unique()))
    # info = pd.read(os.path.join(data_path, file_infoname))

    # X_cat, X_num = preprocessing_cat_data_df(X_cat, X_num, min_size_cat, idx_cat_preprocess, method)


    D = src.Dataset(
        X_num,
        X_cat
    )

    # D.X_cat, D.X_num = preprocessing_cat_data(D.X_cat, D.X_num, min_size_cat, idx_cat_preprocess, method)
    # print(D.X_cat)
    # print(D.X_num["train"]["n_pers_house"])
    # for col in D.X_cat["train"].columns:
    #     print(D.X_cat["train"][col])
    # return

    if change_val:
        D = src.change_val(D)

    # def categorical_to_idx(feature):
    #     unique_categories = np.unique(feature)
    #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
    #     idx_feature = np.array([idx_mapping[category] for category in feature])
    #     return idx_feature

    # for split in ['train', 'val', 'validation']:
    # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

    return src.transform_dataset(D, T, None)


def get_input_embedded_training_data(args):

    embedding_save_path = f'ckpt/{args.folder_save}/train_z.npy' 
    embedding_validation_save_path = f'ckpt/{args.folder_save}/validation_z.npy' 
    
    train_z = torch.tensor(np.load(embedding_save_path)).float()
    validation_z = torch.tensor(np.load(embedding_validation_save_path)).float()

    train_z = train_z[:, 1:, :]
    validation_z = validation_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    V, _, _ = validation_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim) # converted into a vector
    validation_z = validation_z.view(V, in_dim) # converted into a vector

    return train_z, validation_z #, curr_dir, dataset_dir, ckpt_dir, info


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
                
def transform_to_numeric_values(data: np.ndarray, idx_num: np.ndarray, idx_cat: np.ndarray):
    X_num = data[:,idx_num].astype(float)
    
    X_cat = np.zeros(data[:,idx_cat].shape,dtype=int)
    
    categories = []
    for j,id in enumerate(idx_cat):
        unique_values = np.unique(data[:,id])
        categories.append(len(unique_values))
        translation = {}
        for i,a in enumerate(unique_values):
            translation[a] = i
        X_cat[:,j] = np.vectorize(translation.get)(data[:,id])
        
    return X_num, X_cat, categories

def transform_num_into_quantiles(data: pd.DataFrame, name_num: np.ndarray):
    dict_translation_inverse = []
    for name in name_num:
        values = data[name].unique()
        quantiles = (np.concatenate([[0],data[name].value_counts().sort_index().cumsum().to_numpy()])[:-1]+1)/len(data)
        dict_trans = {v:q for v,q in zip(values,quantiles)}
        dict_trans_inv = {q:v for v,q in zip(values,quantiles)}
        data[name] = np.vectorize(dict_trans.get)(data[name])
        dict_translation_inverse.append(dict_trans_inv)
    return data,dict_translation_inverse

def transform_num_into_bins(data: pd.DataFrame, name_num: np.ndarray, n_bins:int):
    for name in name_num:
        bins = data[name]//(1/n_bins)*(1/n_bins)
        for bin in np.unique(bins):
            idx_bin = bins==bin
            data.loc[idx_bin,name] = np.min(data.loc[idx_bin,name])
    return data