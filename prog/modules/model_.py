import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class FOMOnet(nn.Module):

    def __init__(self, enc_chs=(4,32,64,128,256,512,1024), dec_chs=(1024,512,256,128,64,32)):
        super().__init__()

        self.enc_chs = enc_chs
        self.dec_chs = dec_chs

    def encoder(self, x):
        maxpool = nn.MaxPool1d(kernel_size=2)
        init_shape = x.shape[2]
        enc_ftrs = []
        for i, ch in enumerate(self.enc_chs):
            if ch != self.enc_chs[-1]:
                conv_block = self.conv_block(self.enc_chs[i], self.enc_chs[i+1])
                res_block = self.res_block(self.enc_chs[i], self.enc_chs[i+1])
                conv_block, res_block = self.conv_block(x), self.res_block(x)
                conv_block += res_block
                enc_ftrs.append(conv_block)
                x = maxpool(conv_block)
            else:
                conv_block = self.conv_block(self.enc_chs[i], self.enc_chs[i])
                res_block = self.res_block(self.enc_chs[i], self.enc_chs[i])
                conv_block, res_block = self.conv_block(x), self.res_block(x)
                x = conv_block + res_block
        return x, enc_ftrs, init_shape

    def decoder(self, x, enc_ftrs, init_shape):
        for i, ch in enumerate(self.dec_chs):
            if ch != self.dec_chs[-1]:
                transpose = nn.ConvTranspose1d(in_channels=self.dec_chs[i], out_channels=self.dec_chs[i+1], kernel_size=2, stride=2)
                transpose_ = transpose(x)
                cropped = self.crop(transpose_, enc_ftrs[i])
                cat = torch.cat((transpose_, cropped), 1)
                conv_block = self.conv_block(self.dec_chs[i], self.dec_chs[i+1])
                res_block = self.res_block(self.dec_chs[i], self.dec_chs[i+1])
                conv_block, res_block = self.conv_block(cat), self.res_block(cat)
                x = conv_block + res_block
            else:
                x = self.final_block(x)
                out = F.interpolate(x, init_shape)
        return out

    def forward(self, x):
        sigmoid = nn.Sigmoid()
        x, enc_ftrs, init_shape = self.encoder(x)
        out = self.decoder(x, enc_ftrs, init_shape)
        return sigmoid(out)

    def crop(self, x, enc_ftrs):
        chs, dims = x.shape[1:]
        enc_ftrs = torchvision.transforms.CenterCrop([chs, dims])(enc_ftrs)
        return enc_ftrs

    @staticmethod
    def conv_block(in_channels, out_channels, k=5, p=0.5):
        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=k, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=p),
            nn.Conv1d(out_channels, out_channels, kernel_size=k, groups=out_channels, padding='same'),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=p)
        )
        return block
    
    @staticmethod
    def res_block(in_channels, out_channels, k=5, p=0.5):
        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=k, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=p)
        )
        return block
    
    @staticmethod
    def final_block(in_channels, out_channels, k=1):
        block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
        )
        return block