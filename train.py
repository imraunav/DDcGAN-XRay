import torch
from torch import nn
from torch.optim import Adam, lr_scheduler, SGD, RMSprop
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

from model import Generator, Discriminator
from utils import XRayDataset, TVLoss
import hyperparameters


bce = nn.BCELoss()
mse = nn.MSELoss()
l1 = nn.L1Loss()
tv = TVLoss


def content_loss(real_im, gen_im):
    # may need to play around with this
    # return mse(real_im, gen_im)
    eta = hyperparameters.eta
    # return l1(real_im, gen_im) + eta * tv(real_im - gen_im)
    return l1(real_im, gen_im)


def generator_loss(score):
    loss = bce(torch.ones_like(score), score)
    return loss


def discriminator_loss(score_real, score_gen):
    real_loss = bce(torch.ones_like(score_real), score_real)
    gen_loss = bce(torch.zeros_like(score_gen), score_gen)

    return real_loss + gen_loss


def train(generator, disc_l, disc_h, loader, epochs, device):
    gen_epoch_loss = []
    generator.to(device)
    disc_l.to(device)
    disc_h.to(device)

    lr = hyperparameters.learning_rate_init
    decay_rate = hyperparameters.decay_rate
    gen_opt = Adam(generator.parameters(), lr)
    disc_l_opt = Adam(disc_l.parameters(), lr)
    disc_h_opt = Adam(disc_h.parameters(), lr)
    # gen_opt = RMSprop(generator.parameters(), lr)
    # disc_l_opt = SGD(disc_l.parameters(), lr)
    # disc_h_opt = SGD(disc_h.parameters(), lr)
    gen_opt_scheduler = lr_scheduler.ExponentialLR(gen_opt, decay_rate)
    disc_l_opt_scheduler = lr_scheduler.ExponentialLR(disc_l_opt, decay_rate)
    disc_h_opt_scheduler = lr_scheduler.ExponentialLR(disc_h_opt, decay_rate)

    L_gmax = 1000  # just a precaution
    epoch_loss = {"gen": [], "disc_l": [], "disc_h": []}

    disc_l.train()
    disc_h.train()
    generator.train()
    print("Training started...")
    for epoch in range(epochs):
        # set to training mode
        disc_l_runningloss = []
        disc_h_runningloss = []
        gen_runningloss = []

        for bno, batch in enumerate(loader):
            low_imgs, high_imgs = batch
            # send to GPU
            low_imgs = low_imgs.to(device)
            high_imgs = high_imgs.to(device)

            gen_imgs = generator(high_imgs, low_imgs).detach()
            plt.subplot(3, 1, 1)
            plt.imshow(gen_imgs[0].to('cpu')[0], cmap='grey')
            plt.subplot(3, 1, 2)
            plt.imshow(low_imgs[0].to('cpu')[0], cmap='grey')
            plt.subplot(3, 1, 3)
            plt.imshow(high_imgs[0].to('cpu')[0], cmap='grey')
            plt.savefig('./sample_gen.png')
            # Train Discriminators
            # discriminator low energy
            # print("Training Disc l...")
            for _ in range(hyperparameters.I_max):
                real_labels = disc_l(low_imgs)
                gen_labels = disc_l(gen_imgs)
                dloss = discriminator_loss(real_labels, gen_labels)
                disc_l_opt.zero_grad()
                dloss.backward()
                disc_l_opt.step()
                disc_l_runningloss.append(dloss.item())
                print("Discriminator l loss: ", dloss.item())
                if dloss.item() <= hyperparameters.L_max:
                    break

            # discriminator high energy
            # print("Training Disc h...")
            for _ in range(hyperparameters.I_max):
                real_labels = disc_h(high_imgs)
                gen_labels = disc_h(gen_imgs)
                dloss = discriminator_loss(real_labels, gen_labels)
                disc_h_opt.zero_grad()
                dloss.backward()
                disc_h_opt.step()
                disc_h_runningloss.append(dloss.item())
                print("Discriminator h loss: ", dloss.item())
                if dloss.item() <= hyperparameters.L_max:
                    break

            # Train Generator
            # print("Training Gen...")
            def generator_trianer():
                gen_imgs = generator(high_imgs, low_imgs)
                gen_opt.zero_grad()
                cont_loss_l = content_loss(gen_im=gen_imgs, real_im=low_imgs)
                cont_loss_h = content_loss(gen_im=gen_imgs, real_im=high_imgs)
                score_l = disc_l(gen_imgs).detach()
                score_h = disc_h(gen_imgs).detach()
                gen_loss_l = generator_loss(score_l)
                gen_loss_h = generator_loss(score_h)
                gloss = (gen_loss_l + gen_loss_h) + hyperparameters.lam * (
                    cont_loss_l + cont_loss_h
                )
                gloss.backward()
                gen_opt.step()
                print("Generator loss: ", gloss.item())
                # print(f"Scorel: {score_l.shape}, Scoreh: {score_h.shape}, Gloss: {gloss.shape},")
                return (
                    torch.mean(score_l).item(),
                    torch.mean(score_h).item(),
                    gloss.item(),
                )

            for _ in range(hyperparameters.I_max):
                score_l, score_h, gloss = generator_trianer()
                gen_runningloss.append(gloss)
                if score_l >= hyperparameters.L_min or score_h >= hyperparameters.L_min:
                    break
            for _ in range(hyperparameters.I_max):
                _, _, gloss = generator_trianer()
                gen_runningloss.append(gloss)
                if epoch == 0 and bno == 0:  # first loss count
                    L_gmax = 0.8 * gloss
                if gloss < L_gmax:
                    break
            epoch_loss["disc_l"].append(np.mean(disc_l_runningloss))
            epoch_loss["disc_h"].append(np.mean(disc_h_runningloss))
            epoch_loss["gen"].append(np.mean(gen_runningloss))
            print(epoch_loss)

        gen_opt_scheduler.step()
        disc_h_opt_scheduler.step()
        disc_l_opt_scheduler.step()

        # checkpoint
        if (epoch + 1) % hyperparameters.checkpoint_epoch == 0:
            if not os.path.exists("./weights"):
                os.mkdir("./weights")
            torch.save(generator.state_dict(), f"./weights/generator_epoch{epoch+1}.pt")
            torch.save(disc_l.state_dict(), f"./weights/disc_l_epoch{epoch+1}.pt")
            torch.save(disc_h.state_dict(), f"./weights/disc_h_epoch{epoch+1}.pt")

            torch.save(gen_opt.state_dict(), f"./weights/gen_opt_epoch{epoch+1}.pt")
            torch.save(
                disc_l_opt.state_dict(), f"./weights/disc_l_opt_epoch{epoch+1}.pt"
            )
            torch.save(
                disc_h_opt.state_dict(), f"./weights/disc_h_opt_epoch{epoch+1}.pt"
            )
            print(f"Epoch: {epoch+1}")

            if not os.path.exists("./checkpoints"):
                os.mkdir("./checkpoints")
            plt.plot(epoch_loss["gen"], label="Generator loss")
            plt.plot(epoch_loss["disc_l"], label="Disc_l loss")
            plt.plot(epoch_loss["disc_h"], label="Disc_h loss")
            plt.legend()
            plt.savefig("./checkpoints/loss_plot.png")

            # print(low.shape, high.shape)


def main():
    dataset_path = hyperparameters.dataset_path
    batch_size = hyperparameters.batch_size
    num_workers = hyperparameters.num_workers
    epochs = hyperparameters.epochs

    dataset = XRayDataset(dataset_path)
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)
    generator = Generator()  # may need to play around with these
    disc1 = Discriminator()
    disc2 = Discriminator()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = "cpu"
    # device = "mps"
    train(generator, disc1, disc2, loader, epochs, device)


if __name__ == "__main__":
    main()
