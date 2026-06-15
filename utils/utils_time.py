import time
from utils.utils_dir import get_ckpt_dir

def save_time(start_time, config, term):
    ckpt_dir = get_ckpt_dir(config)
    
    path_time = f'{ckpt_dir}\\training_time{term}.txt'
    
    end_time = time.time()
    message = 'Training time: {:.4f} mins'.format((end_time - start_time)/60)
    with open(path_time, "w") as f:
        f.write(message)