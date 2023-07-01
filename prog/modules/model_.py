import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class FOMOnet(nn.Module):

    def __init__(self, enc_chs=(4, 32, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64, 32)):
        super().__init__()

        self.enc_chs = enc_chs
        self.dec_chs = dec_chs

    def encoder(self, x):
        maxpool = nn.MaxPool1d(kernel_size=2)
        init_shape = x.shape[2]
        enc_ftrs = []
        for i, ch in enumerate(self.enc_chs):
            if ch != self.enc_chs[-1]:
                conv_block = self.conv_block(self.enc_chs[i], self.enc_chs[i + 1])
                res_block = self.res_block(self.enc_chs[i + 1], self.enc_chs[i + 1])
                x = conv_block(x)
                x += res_block(x)
                enc_ftrs.append(x)
                x = maxpool(x)
            else:
                conv_block = self.conv_block(self.enc_chs[i], self.enc_chs[i])
                res_block = self.res_block(self.enc_chs[i], self.enc_chs[i])
                x = conv_block(x)
                x += res_block(x)
        return x, enc_ftrs, init_shape

    def decoder(self, x, enc_ftrs, init_shape):
        for i, ch in enumerate(self.dec_chs):
            if ch != self.dec_chs[-1]:
                transpose = nn.ConvTranspose1d(in_channels=self.dec_chs[i], out_channels=self.dec_chs[i + 1],
                                               kernel_size=2, stride=2)
                transpose_ = transpose(x)
                cropped = self.crop(transpose_, enc_ftrs[i])
                cat = torch.cat((transpose_, cropped), 1)
                conv_block = self.conv_block(self.dec_chs[i] + self.dec_chs[i + 1], self.dec_chs[i + 1])
                res_block = self.res_block(self.dec_chs[i + 1], self.dec_chs[i + 1])
                x = conv_block(cat)
                x += res_block(x)
            else:
                x = self.final_block(x)
                out = F.interpolate(x, size=init_shape)
        return out

    def forward(self, x):
        sigmoid = nn.Sigmoid()
        x, enc_ftrs, init_shape = self.encoder(x)
        out = self.decoder(x, enc_ftrs, init_shape)
        return sigmoid(out)

    def crop(self, x, enc_ftrs):
        _, _, dims = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([dims])(enc_ftrs)
        return enc_ftrs

    def conv_block(self, in_channels, out_channels, k=5, p=0.5):
        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=k, groups=in_channels, padding=k // 2),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=p),
            nn.Conv1d(out_channels, out_channels, kernel_size=k, groups=out_channels, padding=k // 2),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=p)
        )
        return block

    def res_block(self, in_channels, out_channels, k=5, p=0.5):
        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=k, groups=in_channels, padding=k // 2),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=p)
        )
        return block

    def final_block(self, in_channels, out_channels, k=1):
        block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
        )
        return block