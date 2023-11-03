import torch
from torch import nn 
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from model import Generator, Discriminator
from utils import XRayDataset
import hyperparameters


EPS = 1e-8 

def gen_loss(D1_fake, D2_fake):
    # Loss for Generator
    G_loss_GAN_D1 = -torch.mean(torch.log(D1_fake + EPS))
    G_loss_GAN_D2 = -torch.mean(torch.log(D2_fake + EPS))
    G_loss_GAN = G_loss_GAN_D1 + G_loss_GAN_D2
    return G_loss_GAN

def train(generator, disc_l, disc_h, loader, epochs, device):
    gen_epoch_loss = []
    lr = hyperparameters.learning_rate
    decay_rate = hyperparameters.decay_rate
    gen_opt = lr_scheduler.ExponentialLR(Adam(generator.parameters(), lr), decay_rate)
    disc_l_opt = lr_scheduler.ExponentialLR(Adam(disc_l.parameters(), lr), decay_rate)
    disc_h_opt = lr_scheduler.ExponentialLR(Adam(disc_h.parameters(), lr), decay_rate)

    adv_crit = nn.BCEloss()

    for epoch in range(epochs):
        # set to training mode
        disc_l.train()
        disc_h.train()
        generator.train()

        print(f"Training epoch: {epoch}")
        for bno, batch in enumerate(loader):
            low_imgs, high_imgs = batch
            # send to GPU
            low_imgs = low_imgs.to(device)
            high_imgs = high_imgs.to(device)

            # Train Discriminators
            disc_l_opt.zero_grad()
            disc_h_opt.zero_grad()

            gen_imgs = generator(low_imgs, high_imgs)
            dl_out_fake = disc_l(gen_imgs.detech()) # avoid backpropagating through generator
            dh_out_fake = disc_h(gen_imgs.detech())

            dl_out_real = disc_l(low_imgs.detech())
            dh_out_real = disc_h(high_imgs.detech())


            # make labels
            fake_labels = torch.zeros(gen_imgs.shape[0], device=device)
            real_labels = torch.ones(gen_imgs.shape[0], device=device)
            # get loss 
            fake_loss_l = adv_crit(dl_out_fake, fake_labels)
            real_loss_l = adv_crit(dl_out_real, real_labels)
            # update discriminators
            dl_loss = fake_loss_l + real_loss_l
            dl_loss.backward()
            disc_l_opt.step()

            fake_loss_h = adv_crit(dh_out_fake, fake_labels)
            real_loss_h = adv_crit(dh_out_real, real_labels)

            dh_loss = fake_loss_h + real_loss_h
            dh_loss.backward()
            disc_h_opt.step()


            # Train Generator
            gen_opt.zero_grad()
            fake_labels = torch.ones(gen_imgs.shape[0], device=device)
            dl_out_fake = disc_l(gen_imgs)
            dh_out_fake = disc_h(gen_imgs)
            




            # print(low.shape, high.shape)


def main():
    dataset_path = hyperparameters.dataset_path
    batch_size = hyperparameters.batch_size
    num_workers = hyperparameters.num_workers
    epochs = hyperparameters.epochs

    dataset = XRayDataset(dataset_path)
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)
    generator = Generator()
    disc1 = Discriminator()
    disc2 = Discriminator()
    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    device = "mps"
    train(generator, disc1, disc2, loader, epochs, device)


if __name__ == "__main__":
    main()
