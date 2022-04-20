import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=kernel_size//2, padding_mode='reflect')

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='nearest'),
            ConvLayer(in_channels, out_channels, kernel_size, stride=1)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, kernel_size)
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv2 = ConvLayer(in_channels, in_channels, kernel_size)
        self.norm2 = nn.InstanceNorm2d(in_channels, affine=True)
        self.ide = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.ide(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        out = x + res
        return self.relu(out)


class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        # encoder
        self.d1 = ConvLayer(3, 32, kernel_size=9)
        self.n1 = nn.InstanceNorm2d(32, affine=True)
        self.d2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.n2 = nn.InstanceNorm2d(64, affine=True)
        self.d3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.n3 = nn.InstanceNorm2d(128, affine=True)
        # residual
        self.r1 = ResidualBlock(128)
        self.r2 = ResidualBlock(128)
        self.r3 = ResidualBlock(128)
        self.r4 = ResidualBlock(128)
        self.r5 = ResidualBlock(128)
        # decoder
        self.u1 = UpsampleBlock(128, 64, kernel_size=3)
        self.n4 = nn.InstanceNorm2d(64, affine=True)
        self.u2 = UpsampleBlock(64, 32, kernel_size=3)
        self.n5 = nn.InstanceNorm2d(32, affine=True)
        self.conv = ConvLayer(32, 3, kernel_size=9)

    def forward(self, x):
        # encoder
        y = self.relu(self.n1(self.d1(x)))
        y = self.relu(self.n2(self.d2(y)))
        y = self.relu(self.n3(self.d3(y)))
        # residual
        y = self.r1(y)
        y = self.r2(y)
        y = self.r3(y)
        y = self.r4(y)
        y = self.r5(y)
        # decoder
        y = self.relu(self.n4(self.u1(y)))
        y = self.relu(self.n5(self.u2(y)))
        return self.conv(y)


def test():
    model = TransformNet()
    x = torch.randn(4, 3, 512, 512)
    out = model(x)
    print(out.shape)


if __name__ == '__main__':
    test()
