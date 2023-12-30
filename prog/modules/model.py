import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class FOMOnet(nn.Module):

    def __init__(self, k=5, p=0.5):
        super().__init__()

        #encoder pooling operation
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=p)
        #encoder convolutional blocks
        self.conv1 = self.conv_block(4, 32, k=k)
        self.conv2 = self.conv_block(32, 64, k=k)
        self.conv3 = self.conv_block(64, 128, k=k)
        self.conv4 = self.conv_block(128, 256, k=k)
        self.conv5 = self.conv_block(256, 512, k=k)
        self.conv6 = self.conv_block(512, 1024, k=k)
        self.convb = self.conv_block(1024, 1024, k=k)
        #decoder convolutional blocks
        self.dconv5 = self.conv_block(1024, 512, k=k)
        self.dconv4 = self.conv_block(512, 256, k=k)
        self.dconv3 = self.conv_block(256, 128, k=k)
        self.dconv2 = self.conv_block(128, 64, k=k)
        self.dconv1 = self.conv_block(64, 32, k=k)
        self.dconvf = self.final_block(32, 1, k=k)
        #decoder upsampling operations
        self.upsample5 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        #output function
        self.sigmoid = nn.Sigmoid()

    def crop(self, x, enc_ftrs):
        chs, dims = x.shape[1:]
        enc_ftrs = torchvision.transforms.CenterCrop([chs, dims])(enc_ftrs)
        return enc_ftrs

    def forward(self, x):
        init_shape = x.shape[2]
        #encoder layer 1
        block1 = self.conv1(x)
        e1_shape = x.shape
        x = self.maxpool(block1)
        x = self.dropout(x)
        
        #encoder layer 2
        block2 = self.conv2(x) 
        e2_shape = x.shape
        x = self.maxpool(block2)
        x = self.dropout(x)

        #encoder layer 3
        block3 = self.conv3(x) 
        e3_shape = x.shape
        x = self.maxpool(block3)
        x = self.dropout(x)

        #encoder layer 4
        block4 = self.conv4(x) 
        e4_shape = x.shape
        x = self.maxpool(block4)
        x = self.dropout(x)

        #encoder layer 5
        block5 = self.conv5(x) 
        e5_shape = x.shape
        x = self.maxpool(block5)
        x = self.dropout(x)

        #encoder layer 6
        block6 = self.conv6(x) 
        x = self.dropout(block6)

        #encoder bottleneck layer
        block7 = self.convb(x) 
        x = self.dropout(block7)

        #decoder layer 5
        upsamp5 = self.upsample5(x)
        upsamp5 = F.interpolate(upsamp5, e5_shape[2])
        cropped5 = self.crop(upsamp5, block5)
        cat5 = torch.cat((upsamp5, cropped5), 1)
        x = self.dconv5(cat5)
        x = self.dropout(x)

        #decoder layer 4
        upsamp4 = self.upsample4(x)
        upsamp4 = F.interpolate(upsamp4, e4_shape[2])
        cropped4 = self.crop(upsamp4, block4)
        cat4 = torch.cat((upsamp4, cropped4), 1)
        x = self.dconv4(cat4)
        x = self.dropout(x)
        
        #decoder layer 3
        upsamp3 = self.upsample3(x)
        upsamp3 = F.interpolate(upsamp3, e3_shape[2])
        cropped3 = self.crop(upsamp3, block3)
        cat3 = torch.cat((upsamp3, cropped3), 1)
        x = self.dconv3(cat3)
        x = self.dropout(x)

        #decoder layer 2
        upsamp2 = self.upsample2(x)
        upsamp2 = F.interpolate(upsamp2, e2_shape[2])
        cropped2 = self.crop(upsamp2, block2)
        cat2 = torch.cat((upsamp2, cropped2), 1)
        x = self.dconv2(cat2)
        x = self.dropout(x)

        #decoder layer 1
        upsamp1 = self.upsample1(x)
        upsamp1 = F.interpolate(upsamp1, e1_shape[2])
        cropped1 = self.crop(upsamp1, block1)
        cat1 = torch.cat((upsamp1, cropped1), 1)
        x = self.dconv1(cat1)
        x = self.dropout(x)

        #decoder layer f (final layer)
        x = self.dconvf(x)
        x = F.interpolate(x, init_shape)
        return self.sigmoid(x)

    @staticmethod
    def conv_block(in_channels, out_channels, k=5):
        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=k, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=k, groups=in_channels, padding='same'),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=k, groups=out_channels, padding='same'),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
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