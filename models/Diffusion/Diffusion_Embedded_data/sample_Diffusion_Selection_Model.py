import torch
import numpy as np
from utils import get_input_train
from model_Diffusion import MLPDiffusion, Model_Diffusion
from tqdm import tqdm

from utils.utils_diffusion import sample
from tqdm import tqdm


def sample_Diffusion_from_Transformer_data(args, projection_dim = 1024):
    if ("term" in args.__dict__.keys()):
        term = args.term
    else:
        term = f"_{projection_dim}"
        
    name_model = "model_diffusion"


    device = args.device
    ref_dir= args.reference_dir

    path_model = f'{ref_dir}/ckpt/{args.folder_save}'

    ref_dir= args.reference_dir

    train_z = get_input_train(args)[0]

    in_dim = train_z.shape[1] 

    mean_train, std = train_z.mean(0), train_z.std(0)


    #Diffusion model
    import_model = torch.load(f'{path_model}/{name_model}{term}.pt')

    denoise_fn = MLPDiffusion(in_dim, projection_dim).to(device)

    model = Model_Diffusion(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    model.load_state_dict(import_model)

    model.to(device)

    model.eval()

    n_batch = args.batch_size_generation

    tot= 0
    res = []
    n = int(np.ceil(args.n_generation/n_batch))

    if ("num_steps" in args.__dict__.keys()):
        num_steps = args.num_steps
    else:
        num_steps = 500
    for _ in tqdm(range(n)):
        n_samples = min(args.n_generation-tot, n_batch)
        x = sample(model.denoise_fn_D, n_samples, in_dim, num_steps)
        tot+= n_samples
        res.append((x.cpu()*2 +mean_train).numpy())
    res = np.concatenate(res)

    np.save(f'{ref_dir}/ckpt/{args.folder_save}/encoded_generated{term}.npy', res)