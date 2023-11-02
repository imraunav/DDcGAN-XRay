from torch.utils.data import DataLoader

from model import Generator, Discriminator
from utils import XRayDataset
import hyperparameters

def train(generator, disc1, disc2, loader):

    #complete this loop for training a gan based on the algorithm

    for low, high in loader:
        pass
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


if __name__ == "__main__":
    main()
