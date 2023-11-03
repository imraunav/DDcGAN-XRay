import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, le, he):
        img = torch.concat([le, he], dim=-3)
        # print("le", le.shape)
        # print("he", he.shape)
        # print("img", img.shape)
        code = self.encoder(img)
        gen = self.decoder(code)
        return gen


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Using same padding instead of reflective padding. Should I do reflective padding?
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=48,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(48),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=48,
                out_channels=48,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(48),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=48,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(48),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=144,
                out_channels=48,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(48),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=192,
                out_channels=48,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(48),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        x2 = F.relu(x2)
        x2 = torch.concat([x1, x2], dim=-3)

        x3 = self.conv3(x2)
        x3 = F.relu(x3)
        x3 = torch.concat([x2, x3], dim=-3)

        x4 = self.conv4(x3)
        x4 = F.relu(x4)
        x4 = torch.concat([x3, x4], dim=-3)

        x5 = self.conv5(x4)
        x5 = F.relu(x5)
        x5 = torch.concat([x4, x5], dim=-3)

        return x5


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=240,
                out_channels=240,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.BatchNorm2d(240),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=240,
                out_channels=128,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.BatchNorm2d(64),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.BatchNorm2d(32),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        x2 = F.relu(x2)

        x3 = self.conv3(x2)
        x3 = F.relu(x3)

        x4 = self.conv4(x3)
        x4 = F.relu(x4)

        x5 = self.conv5(x4)
        x5 = F.tanh(x5)
        x5 = (x5 / 2) + 0.5  # bring back to range [0, 1]
        # print('Decoder out: ', x5.shape)
        return x5


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding="valid",
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding="valid",
            ),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding="valid",
            ),
            nn.BatchNorm2d(64),
        )

        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features=9 * 9 * 64, out_features=1)
        )

    def forward(self, x):
        # print("Discriminator in: ", x.shape)
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.fc(x)
        x = F.tanh(x)

        x = (x / 2) + 0.5

        return x
