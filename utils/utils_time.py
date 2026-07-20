import time
import os
from utils.utils_dir import get_ckpt_dir

def save_time(start_time, config, term):
    ckpt_dir = get_ckpt_dir(config)
    if (not os.path.exists(ckpt_dir)):
        os.makedirs(ckpt_dir)
    path_time = os.path.join(ckpt_dir, "training_time" + term + ".txt")
    
    end_time = time.time()
    message = 'Training time: {:.4f} mins'.format((end_time - start_time)/60)
    with open(path_time, "w") as f:
        f.write(message)