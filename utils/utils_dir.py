import os

def get_info_path(config):
    datapath = "Data"
    dataname = config.dataname
    infoname = config.infoname
    return f'{datapath}\\{dataname}\\{infoname}'


def get_data_dir(config):
    datapath = "Data"
    dataname = config.dataname
    return f'{datapath}\\{dataname}\\{config.data_folder_size}'


def get_testing_data_dir(config):
    datapath = "Data"
    dataname = config.dataset_evaluation
    dataname_evaluation = config.dataname
    if(dataname==dataname_evaluation):
        return f'{datapath}\\{dataname}\\{config.data_folder_size}'
    else:
        return f'{datapath}\\{dataname_evaluation}'

def get_ckpt_dir(config):
    return f'ckpt\\{config.folder_save}\\{config.size_data_str}' 

def get_folder_sampling(config, term):
    res = f'{config.sample_folder}\\{config.variable}\\{config.size_data_str}\\{config.folder_save+term}'
    if not (os.path.isdir(res)):
        os.makedirs(res)
    return res

def get_folder_evaluation(config, term):
    folder_sampling = get_folder_sampling(config, term)
    
    folder_evaluation = f"{folder_sampling}\\Evaluation_on_{config.dataset_evaluation}"
    if not (os.path.isdir(folder_evaluation)):
        os.makedirs(folder_evaluation)
        
    return folder_evaluation

 

def get_filename_sampling(config):
    n_sample = config.n_generation
    filename_sampling = f"generated_population_{n_sample}"
    return filename_sampling

def get_file_path_sampling(config, term):
    filename_sampling = get_filename_sampling(config)
    folder_sampling = get_folder_sampling(config, term)
    return f'{folder_sampling}\\{filename_sampling}.csv'


def get_encoded_filename(config, term):
    path_model_save = get_ckpt_dir(config)
    if (not os.path.exists(path_model_save)):
        os.makedirs(path_model_save)
    return f'{path_model_save}\\encoded_generated{term}.npy'


def get_model_torch_path(config, name_model, term):
    path_model_save = get_ckpt_dir(config)
    if (not os.path.exists(path_model_save)):
        os.makedirs(path_model_save)
    return f'{path_model_save}\\{name_model}_{term}.pt'
