import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


# U-Net
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()

        self.c1 = ConvBlock(in_channels, base_filters)
        self.p1 = nn.MaxPool2d(2)

        self.c2 = ConvBlock(base_filters, base_filters * 2)
        self.p2 = nn.MaxPool2d(2)

        self.c3 = ConvBlock(base_filters * 2, base_filters * 4, dropout=0.2)
        self.p3 = nn.MaxPool2d(2)

        self.c4 = ConvBlock(base_filters * 4, base_filters * 8, dropout=0.3)
        self.p4 = nn.MaxPool2d(2)

        self.c5 = ConvBlock(base_filters * 8, base_filters * 16, dropout=0.5)

        # Decoder
        self.up6 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.c6 = ConvBlock(base_filters * 16, base_filters * 8)

        self.up7 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.c7 = ConvBlock(base_filters * 8, base_filters * 4)

        self.up8 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.c8 = ConvBlock(base_filters * 4, base_filters * 2)

        self.up9 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.c9 = ConvBlock(base_filters * 2, base_filters)

        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.p1(c1)

        c2 = self.c2(p1)
        p2 = self.p2(c2)

        c3 = self.c3(p2)
        p3 = self.p3(c3)

        c4 = self.c4(p3)
        p4 = self.p4(c4)

        c5 = self.c5(p4)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)

        out = torch.sigmoid(self.out_conv(c9))
        return out

