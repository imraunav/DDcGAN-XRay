import os
import torch
from torch import nn, optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import (
    DistributedSampler,
)  # Distribute data across multiple gpus
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
import numpy as np

import hyperparameters
from utils import XRayDataset
from model import Generator, Discriminator


def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "54321"  # select any idle port on your machine

    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def dataloader_ddp(
    trainset: Dataset,
    bs: int,
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    sampler_train = DistributedSampler(trainset, shuffle=True)
    trainloader = DataLoader(
        trainset,
        batch_size=bs,
        shuffle=False,
        sampler=sampler_train,
        num_workers=hyperparameters.num_workers,
    )

    return trainloader, sampler_train


class TrainerDDP:
    def __init__(
        self,
        gpu_id: int,
        generator: nn.Module,
        disc_l: nn.Module,
        disc_h: nn.Module,
        trainloader: DataLoader,
        sampler_train: DistributedSampler,
    ) -> None:
        print("Initializing trainer on GPU:", gpu_id)
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.gpu_id = gpu_id
        self.generator = DDP(generator, device_ids=[gpu_id], output_device=gpu_id)
        self.disc_l = DDP(disc_l, device_ids=[gpu_id], output_device=gpu_id)
        self.disc_h = DDP(disc_h, device_ids=[gpu_id], output_device=gpu_id)

        self.trainloader = trainloader

        self.sampler_train = sampler_train

        self.lr = hyperparameters.learning_rate_init
        self.decay_rate = hyperparameters.decay_rate
        self.gen_opt = optim.Adam(generator.parameters(), self.lr)
        self.disc_l_opt = optim.Adam(disc_l.parameters(), self.lr)
        self.disc_h_opt = optim.Adam(disc_h.parameters(), self.lr)

        self.gen_opt_scheduler = optim.lr_scheduler.ExponentialLR(self.self.gen_opt, self.decay_rate)
        self.disc_l_opt_scheduler = optim.lr_scheduler.ExponentialLR(
            self.disc_l_opt, self.decay_rate
        )
        self.disc_h_opt_scheduler = optim.lr_scheduler.ExponentialLR(
            self.disc_h_opt, self.decay_rate
        )

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def content_loss(self, real_im, gen_im):
        # may need to play around with this
        # return mse(real_im, gen_im)
        eta = hyperparameters.eta
        # return l1(real_im, gen_im) + eta * tv(real_im - gen_im)
        return self.l1(real_im, gen_im)

    def generator_loss(self, score):
        loss = self.bce(torch.ones_like(score), score)
        return loss

    def discriminator_loss(self, score_real, score_gen):
        real_loss = self.bce(torch.ones_like(score_real), score_real)
        gen_loss = self.bce(torch.zeros_like(score_gen), score_gen)

        return real_loss + gen_loss

    def _save_checkpoint(self, epoch: int):
        ckp = self.generator.module.state_dict()
        model_path = f"./weights/generator_{epoch}.pt"
        torch.save(ckp, model_path)

        ckp = self.disc_l.module.state_dict()
        model_path = f"./weights/disc_l_{epoch}.pt"
        torch.save(ckp, model_path)

        ckp = self.disc_h.module.state_dict()
        model_path = f"./weights/disc_h_{epoch}.pt"
        torch.save(ckp, model_path)

    def train(self, epochs: int):
        self.disc_l.train()
        self.disc_h.train()
        self.generator.train()

        L_gmax = 1000  # just a precaution
        epoch_loss = {"gen": [], "disc_l": [], "disc_h": []}

        for epoch in range(epochs):
            disc_l_runningloss = []
            disc_h_runningloss = []
            gen_runningloss = []
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            self.sampler_train.set_epoch(epoch)
            for bno, batch in enumerate(self.trainloader):
                low_imgs, high_imgs = batch
                low_imgs = low_imgs.to(self.gpu_id)
                high_imgs = high_imgs.to(self.gpu_id)

                gen_imgs = self.generator(
                    high_imgs, low_imgs
                ).detach()  # don't track grads here

                # Train Discriminators
                # discriminator low energy
                for _ in range(hyperparameters.I_max):
                    # with torch.autocast(device_type=device, dtype=torch.float64):
                    real_labels = self.disc_l(low_imgs)
                    gen_labels = self.disc_l(gen_imgs)
                    dloss = self.discriminator_loss(real_labels, gen_labels)
                    if hyperparameters.debug:
                        print(
                            "Epoch :",
                            epoch,
                            "GPU : ",
                            self.gpu_id,
                            "dloss l:",
                            dloss.item(),
                        )
                    self.disc_l_opt.zero_grad()
                    dloss.backward()
                    # scaler.scale(dloss).backward()
                    self.disc_l_opt.step()
                    # scaler.step(disc_l_opt)
                    # scaler.update()
                    disc_l_runningloss.append(dloss.item())
                    if hyperparameters.debug:
                        print("Discriminator l loss: ", dloss.item())
                    if dloss.item() <= hyperparameters.L_max:
                        break

                # discriminator high energy
                for _ in range(hyperparameters.I_max):
                    # with torch.autocast(device_type=device, dtype=torch.float64):
                    real_labels = self.disc_h(low_imgs)
                    gen_labels = self.disc_h(gen_imgs)
                    dloss = self.discriminator_loss(real_labels, gen_labels)
                    if hyperparameters.debug:
                        print(
                            "Epoch :",
                            epoch,
                            "GPU : ",
                            self.gpu_id,
                            "dloss h:",
                            dloss.item(),
                        )
                    self.disc_h_opt.zero_grad()
                    dloss.backward()
                    # scaler.scale(dloss).backward()
                    self.disc_h_opt.step()
                    # scaler.step(disc_l_opt)
                    # scaler.update()
                    disc_h_runningloss.append(dloss.item())
                    if hyperparameters.debug:
                        print("Discriminator h loss: ", dloss.item())
                    if dloss.item() <= hyperparameters.L_max:
                        break

                # Train Generator
                def generator_trianer():
                    # with torch.autocast(device_type=device, dtype=torch.float64):
                    gen_imgs = self.generator(high_imgs, low_imgs)
                    self.gen_opt.zero_grad()
                    cont_loss_l = self.content_loss(gen_im=gen_imgs, real_im=low_imgs)
                    cont_loss_h = self.content_loss(gen_im=gen_imgs, real_im=high_imgs)
                    # with torch.autocast(device_type=device, dtype=torch.float64):
                    score_l = self.disc_l(gen_imgs).detach()
                    score_h = self.disc_h(gen_imgs).detach()
                    gen_loss_l = self.generator_loss(score_l)
                    gen_loss_h = self.generator_loss(score_h)
                    if hyperparameters.debug:
                        print(
                            "Epoch :",
                            epoch,
                            "GPU : ",
                            self.gpu_id,
                            "gloss:",
                            f"{gen_loss_l.item():.4f}",
                            f"{gen_loss_h.item():.4f}",
                            "cont loss:",
                            f"{cont_loss_l.item():.4f}",
                            f"{cont_loss_h.item():.4f}",
                        )
                    gloss = (gen_loss_l + gen_loss_h) + hyperparameters.lam * (
                        cont_loss_l + cont_loss_h
                    )
                    # scaler.scale(gloss).backward()
                    gloss.backward()
                    self.gen_opt.step()
                    # scaler.step(gen_opt)
                    # scaler.update()
                    if hyperparameters.debug:
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
                    if (
                        score_l >= hyperparameters.L_min
                        or score_h >= hyperparameters.L_min
                    ):
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
                if hyperparameters.debug:
                    print(epoch_loss)
            self.gen_opt_scheduler.step()
            self.disc_h_opt_scheduler.step()
            self.disc_l_opt_scheduler.step()
            # only save once on master gpu
            if self.gpu_id == 0 and epoch % hyperparameters.save_every == 0:
                self._save_checkpoint(epoch)
        # save last epoch
        self._save_checkpoint(epochs - 1)


def main(rank, world_size):
    ddp_setup(rank, world_size)  # initialize ddp

    train_dataset = XRayDataset(hyperparameters.dataset_path)
    train_dataloader, train_sampler = dataloader_ddp(
        train_dataset, hyperparameters.batch_size
    )
    generator = Generator().cuda(rank)
    discriminator_l = Discriminator().cuda(rank)
    discriminator_h = Discriminator().cuda(rank)

    trainer = TrainerDDP(
        gpu_id=rank,
        generator=generator,
        disc_l=discriminator_l,
        disc_h=discriminator_h,
        trainloader=train_dataloader,
        sampler_train=train_sampler,
    )
    trainer.train(hyperparameters.epochs)
    destroy_process_group()  # clean up


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPU.")
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )  # nprocs - total number of processes - # gpus
