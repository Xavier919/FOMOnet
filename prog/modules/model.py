import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class FOMOnet(nn.Module):

    def __init__(self, num_channels=4):
        super().__init__()

        #encoder pooling operation
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        #encoder convolutional blocks
        self.conv1 = self.conv_block(num_channels, 32)
        self.conv2 = self.conv_block(32, 64)
        self.conv3 = self.conv_block(64, 128)
        self.conv4 = self.conv_block(128, 256)
        self.conv5 = self.conv_block(256, 512)
        self.conv6 = self.conv_block(512, 1024)
        self.conv7 = self.conv_block(1024, 2048)
        self.convbot = self.conv_block(2048, 2048)
        #encoder residual blocks
        self.res1 = self.res_block(num_channels, 32)
        self.res2 = self.res_block(32, 64)
        self.res3 = self.res_block(64, 128)
        self.res4 = self.res_block(128, 256)
        self.res5 = self.res_block(256, 512)
        self.res6 = self.res_block(512, 1024)
        self.res7 = self.res_block(1024, 2048)
        self.resbot = self.res_block(2048, 2048)
        #decoder convolutional blocks
        self.dconv7 = self.conv_block(2048, 1024)
        self.dconv6 = self.conv_block(1024, 512)
        self.dconv5 = self.conv_block(512, 256)
        self.dconv4 = self.conv_block(256, 128)
        self.dconv3 = self.conv_block(128, 64)
        self.dconv2 = self.conv_block(64, 32)
        self.dconv1 = self.final_block(32, 1)
        #decoder residual blocks
        self.dres7 = self.res_block(2048, 1024)
        self.dres6 = self.res_block(1024, 512)
        self.dres5 = self.res_block(512, 256)
        self.dres4 = self.res_block(256, 128)
        self.dres3 = self.res_block(128, 64)
        self.dres2 = self.res_block(64, 32)
        #decoder upsampling operations
        self.upsample7 = nn.ConvTranspose1d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2)
        self.upsample6 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.upsample5 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        #output function
        self.sigmoid = nn.Sigmoid()

    def crop(self, x, enc_ftrs):
        chs, dims = x.shape[1:]
        enc_ftrs = torchvision.transforms.CenterCrop([chs, dims])(enc_ftrs)
        return enc_ftrs

    def forward(self, x):
        init_shape = x.shape[2]
        #encoder layer 1
        block1 = self.conv1(x) + self.res1(x)
        x = self.maxpool(block1)
        #encoder layer 2
        block2 = self.conv2(x) + self.res2(x)
        x = self.maxpool(block2)
        #encoder layer 3
        block3 = self.conv3(x) + self.res3(x)
        x = self.maxpool(block3)
        #encoder layer 4
        block4 = self.conv4(x) + self.res4(x)
        x = self.maxpool(block4)
        #encoder layer 5
        block5 = self.conv5(x) + self.res5(x)
        x = self.maxpool(block5)
        #encoder layer 6
        block6 = self.conv6(x) + self.res6(x)
        x = self.maxpool(block6)
        #encoder layer 7
        block7 = self.conv7(x) + self.res7(x)
        x = self.maxpool(block7)
        #bottleneck layer
        bottleneck = self.convbot(x) + self.resbot(x)
        #decoder layer 7
        upsamp7 = self.upsample7(bottleneck)
        cropped7 = self.crop(upsamp7, block7)
        cat7 = torch.cat((upsamp7, cropped7), 1)
        x = self.dconv7(cat7) + self.dres7(cat7)
        #decoder layer 6
        upsamp6 = self.upsample6(x)
        cropped6 = self.crop(upsamp6, block6)
        cat6 = torch.cat((upsamp6, cropped6), 1)
        x = self.dconv6(cat6) + self.dres6(cat6)
        #decoder layer 5
        upsamp5 = self.upsample5(x)
        cropped5 = self.crop(upsamp5, block5)
        cat5 = torch.cat((upsamp5, cropped5), 1)
        x = self.dconv5(cat5) + self.dres5(cat5)
        #decoder layer 4
        upsamp4 = self.upsample4(x)
        cropped4 = self.crop(upsamp4, block4)
        cat4 = torch.cat((upsamp4, cropped4), 1)
        x = self.dconv4(cat4) + self.dres4(cat4)
        #decoder layer 3
        upsamp3 = self.upsample3(x)
        cropped3 = self.crop(upsamp3, block3)
        cat3 = torch.cat((upsamp3, cropped3), 1)
        x = self.dconv3(cat3) + self.dres3(cat3)
        #decoder layer 2
        upsamp2 = self.upsample2(x)
        cropped2 = self.crop(upsamp2, block2)
        cat2 = torch.cat((upsamp2, cropped2), 1)
        x = self.dconv2(cat2) + self.dres2(cat2)
        #decoder layer 1 (final layer)
        out = self.dconv1(x)
        out = F.interpolate(out, init_shape)

        return self.sigmoid(out)

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
            #nn.Conv1d(in_channels, in_channels, kernel_size=k, groups=in_channels, padding='same'),
            #nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding='same'),
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
            nn.BatchNorm1d(out_channels)
        )
        return block