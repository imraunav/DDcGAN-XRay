import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
    return l1(real_im, gen_im) + eta * tv(real_im - gen_im)


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

    lr = hyperparameters.learning_rate
    decay_rate = hyperparameters.decay_rate
    gen_opt = lr_scheduler.ExponentialLR(Adam(generator.parameters(), lr), decay_rate)
    disc_l_opt = lr_scheduler.ExponentialLR(Adam(disc_l.parameters(), lr), decay_rate)
    disc_h_opt = lr_scheduler.ExponentialLR(Adam(disc_h.parameters(), lr), decay_rate)
    L_gmax = 1000  # just a precaution
    epoch_loss = {"gen": [], "disc_l": [], "disc_h": []}
    for epoch in range(epochs):
        # set to training mode
        disc_l.train()
        disc_h.train()
        generator.train()

        for bno, batch in enumerate(loader):
            low_imgs, high_imgs = batch
            # send to GPU
            low_imgs = low_imgs.to(device)
            high_imgs = high_imgs.to(device)

            gen_imgs = generator(high_imgs, low_imgs)

            # Train Discriminators
            # discriminator low energy
            for _ in range(hyperparameters.I_max):
                disc_l_opt.zero_grad()
                real_lables = disc_l(low_imgs)
                gen_lables = disc_l(gen_imgs)
                dloss = discriminator_loss(real_lables, gen_lables)
                dloss.backwards()
                disc_l_opt.step()
                if dloss.detach() <= hyperparameters.L_max:
                    break
            epoch_loss["disc_l"].append(dloss)

            # discriminator high energy
            for _ in range(hyperparameters.I_max):
                disc_h_opt.zero_grad()
                real_lables = disc_h(high_imgs)
                gen_lables = disc_h(gen_imgs)
                dloss = discriminator_loss(real_lables, gen_lables)
                dloss.backwards()
                disc_h_opt.step()
                if dloss.detach() <= hyperparameters.L_max:
                    break
            epoch_loss["disc_h"].append(dloss)

            # Train Generator
            def generator_trianer():
                gen_imgs = generator(high_imgs, low_imgs)
                gen_opt.zero_grad()
                cont_loss_l = content_loss(gen_im=gen_imgs, real_im=low_imgs)
                cont_loss_h = content_loss(gen_im=gen_imgs, real_im=high_imgs)
                score_l = disc_l(gen_imgs)
                score_h = disc_h(gen_imgs)
                gen_loss_l = generator_loss(score_l)
                gen_loss_h = generator_loss(score_h)
                gloss = (gen_loss_l + gen_loss_h) + hyperparameters.lam * (
                    cont_loss_l + cont_loss_h
                )
                gloss.backwards()
                gen_opt.step()
                return score_l.detach(), score_h.detach(), gloss.detach()

            for _ in range(hyperparameters.I_max):
                score_l, score_h, _ = generator_trianer()
                if score_l >= hyperparameters.L_min or score_h >= hyperparameters.L_min:
                    break
            for _ in range(hyperparameters.I_max):
                _, _, gloss = generator_trianer()
                if epoch == 0 and bno == 0:  # first loss count
                    L_gmax = 0.8 * gloss
                if gloss < L_gmax:
                    break
            epoch_loss["gen"].append(gloss)

        # checkpoint
        if (epoch + 1) % hyperparameters.checkpoint_epoch == 0:
            torch.save(generator.state_dict(), f"./weights/generator_epoch{epoch+1}.pt")
            torch.save(disc_l.state_dict(), f"./weights/disc_l_epoch{epoch+1}.pt")
            torch.save(disc_h.state_dict(), f"./weights/disc_h_epoch{epoch+1}.pt")

            torch.save(gen_opt.state_dict(), f"./weights/gen_opt_epoch{epoch+1}.pt")
            torch.save(disc_l_opt.state_dict(), f"./weights/disc_l_opt_epoch{epoch+1}.pt")
            torch.save(disc_h_opt.state_dict(), f"./weights/disc_h_opt_epoch{epoch+1}.pt")
            print(f"Epoch: {epoch+1}")

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
    generator = nn.DataParallel(Generator()) # may need to play around with these
    disc1 = nn.DataParallel(Discriminator())
    disc2 = nn.DataParallel(Discriminator())
    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    device = "mps"
    train(generator, disc1, disc2, loader, epochs, device)


if __name__ == "__main__":
    main()
