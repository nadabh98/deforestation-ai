import torch
import torch.nn as nn


# ============================================================
# BLOC DOUBLE CONVOLUTION
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(self, in_channels=8, out_channels=1):

        super().__init__()

        # ----------------------------------------------------
        # ENCODEUR
        # ----------------------------------------------------

        self.encoder1 = DoubleConv(
            in_channels,
            32
        )

        self.encoder2 = DoubleConv(
            32,
            64
        )

        self.encoder3 = DoubleConv(
            64,
            128
        )

        self.encoder4 = DoubleConv(
            128,
            256
        )

        # ----------------------------------------------------
        # BOTTLENECK
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(
            256,
            512
        )

        # ----------------------------------------------------
        # POOLING
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.decoder4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.decoder3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = DoubleConv(
            64,
            32
        )

        # ----------------------------------------------------
        # SORTIE
        # ----------------------------------------------------

        self.output = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1
        )


    def forward(self, x):

        # ----------------------------------------------------
        # ENCODEUR
        # ----------------------------------------------------

        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool(e1)
        )

        e3 = self.encoder3(
            self.pool(e2)
        )

        e4 = self.encoder4(
            self.pool(e3)
        )

        # ----------------------------------------------------
        # BOTTLENECK
        # ----------------------------------------------------

        b = self.bottleneck(
            self.pool(e4)
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.decoder4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.decoder3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.decoder2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.decoder1(d1)

        # ----------------------------------------------------
        # SORTIE
        # ----------------------------------------------------

        return self.output(d1)
