import torch
import torch.nn as nn
import torch.nn.functional as F



class prelu(nn.Module):
    def __init__(self):
        super(prelu, self).__init__()
        self.p=torch.nn.PReLU()


    def forward(self, x):
        x=self.p(x)
        return x

class tsc(nn.Module):
    def __init__(self, channels):
        super(tsc, self).__init__()
        self.body = nn.Sequential(
            nn.ConvTranspose2d(channels, channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            prelu(),
            nn.Conv2d(channels//2, channels, 3, 1, 1, bias=True),
        )

    def __call__(self, x):
        out = self.body(x)
        return out

class cpc(nn.Module):
    def __init__(self, channels):
        super(cpc, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, 1, 1, bias=True),
            prelu(),
            nn.Conv2d(channels//2, channels, 3, 1, 1, bias=True),
        )

    def __call__(self, x):
        out = self.body(x)
        return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation=None, norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

