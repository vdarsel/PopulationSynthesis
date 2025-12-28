from models.GAN.WGAN_embeded_data.model_WGAN_embedded import Generator, Discriminator, calc_gradient_penalty
import os
from utils.utils_train import get_input_embedded_training_data, set_requires_grad
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np


def train_WGAN_embedding_data(args, term, k=256):
    
    print("\n\nTraining WGAN on embedded data\n\n")

    ##################
    ### Parameters ###
    ##################

    device = args.device

    cktp_dir = 'ckpt'

    path_save = f'{cktp_dir}/{args.folder_save}'
    path_time = f'{cktp_dir}/{args.folder_save}/training_time{term}.txt'

    if (not os.path.exists(path_save)):
        os.makedirs(path_save)

    num_epochs = args.WGAN_embedded.n_epochs
    num_epochs_D = args.WGAN_embedded.n_epochs_Discriminator
    
    #################
    ### Load Data ###
    #################
    
    train_z, validation_z = get_input_embedded_training_data(args)
    train_z = np.concatenate([np.concatenate([train_z,validation_z])])

    in_dim = train_z.shape[1]

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / (2*std) #center dat on 0 with the same range
    train_data = train_z

    batch_size = args.WGAN_embedded.batch_size
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    
    ######################
    ### Initiate Model ###
    ######################

    generator_net = Generator(in_dim,k=k).to(device)
    discriminator_net = Discriminator(in_dim,k=k).to(device)

    num_params_1 = sum(p.numel() for p in generator_net.parameters())
    num_params_2 = sum(p.numel() for p in discriminator_net.parameters())
    print("Number of parameters", num_params_1+num_params_2)


    optimizer_G = torch.optim.Adam(generator_net.parameters(), lr=5e-4, weight_decay=0)

    optimizer_D = torch.optim.Adam(discriminator_net.parameters(), lr=1e-3, weight_decay=0)
    
    ###################
    ### Train Model ###
    ###################

    generator_net.train()

    start_time = time.time()
    generator_net.train()
    discriminator_net.train()

    loss_G_epoch = []
    loss_D_real_epoch = []
    loss_D_fake_epoch = []
    i=0

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        loss_G_train = []
        loss_D_real_train = []
        loss_D_fake_train = []
        loss_D_train = []

        for data in pbar:
            i+=1
            # def should(freq):
            #     return freq > 0 and (i % freq == 0 or i == num_batch_train)
            batch_size_data = len(data)

            real_data = data.to(device)
            input = generator_net.sample_latent(batch_size_data).to(device)

            # forward netG
            # print(input.detach())
            fake_data = generator_net(input)
            # backward discriminator_net
            set_requires_grad(discriminator_net, True)
            optimizer_D.zero_grad()

            pred_real = discriminator_net(real_data)
            pred_fake = discriminator_net(fake_data.detach())

            # GAN Discriminator Loss
            loss_D_real = (torch.mean(pred_real))
            loss_D_fake = (torch.mean(pred_fake))

            # Gradient penalty Loss
            GP_loss = calc_gradient_penalty(discriminator_net, real_data, fake_data, device, lambda_=1)

            # WGAN Discriminator Loss
            loss_D = (loss_D_fake - loss_D_real) + GP_loss

            loss_D.backward()
            optimizer_D.step()

            # backward netG
            if ((i%args.WGAN_embedded.freq_train_G == 0)&(epoch>num_epochs_D)):
                set_requires_grad(discriminator_net, False)
                optimizer_G.zero_grad()

                pred_fake = discriminator_net(fake_data)

                # loss_G = fn_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G = -torch.mean(pred_fake)

                loss_G.backward()
                optimizer_G.step()

                # get losses
                loss_G_train += [loss_G.item()]
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]
            loss_D_train += [loss_D.item()]
        print('epoch: {}, Train loss Discriminator fake data: {:.6f}, Train loss Discriminator real data: {:.6f}, Train loss Discriminator : {:.6f}, Train loss Generator:{:.6f}'.format(epoch, np.mean(loss_D_fake_train), np.mean(loss_D_real_train), np.mean(loss_D_train) ,np.mean(loss_G_train) ))
        loss_G_epoch.append(np.mean(loss_G_train))
        loss_D_real_epoch.append(np.mean(loss_D_real_train))
        loss_D_fake_epoch.append(np.mean(loss_D_fake_train))

    
    ##################
    ### Save Model ###
    ##################

    torch.save(generator_net.state_dict(),f'{path_save}/generator{term}.pt')
    torch.save(discriminator_net.state_dict(),f'{path_save}/discriminator{term}.pt')

    end_time = time.time()
    message = 'Training time: {:.4f} mins'.format((end_time - start_time)/60)
    with open(path_time, "w") as f:
        f.write(message)