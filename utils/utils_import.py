import pandas as pd
import torch

from utils.utils_dir import get_info_path, get_model_torch_path

def get_info_file(config):
    attr_setname = config.attributes_setname

    info_path = get_info_path(config)
    
    info = pd.read_csv(info_path, sep = ";")
    info = info[info[attr_setname]]
    
    return info.reset_index()


def import_data(filename, columns, columns_cat):

    data = pd.read_csv(filename, sep=";", low_memory=False, usecols=columns)
    
    for idx in columns_cat:
        data[idx] = data[idx].astype(str)

    return data


def import_torch_model(config, name_model, term):
    path_model = get_model_torch_path(config, name_model, term)
    model = torch.load(path_model)
    return model
