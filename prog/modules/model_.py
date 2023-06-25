import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class FOMOnet(nn.Module):

    def __init__(self, num_channels=4):
        super().__init__()

        self.maxpool = nn.MaxPool1d(kernel_size=2)
        #encoder
        self.conv_encoder1 = self.conv_block(num_channels, 32)
        self.res_encoder1 = self.res_block(num_channels, 32)
        self.conv_encoder2 = self.conv_block(32, 64)
        self.res_encoder2 = self.res_block(32, 64)
        self.conv_encoder3 = self.conv_block(64, 128)
        self.res_encoder3 = self.res_block(64, 128)
        self.conv_encoder4 = self.conv_block(128, 256)
        self.res_encoder4 = self.res_block(128, 256)
        self.conv_encoder5 = self.conv_block(256, 512)
        self.res_encoder5 = self.res_block(256, 512)
        self.conv_encoder6 = self.conv_block(512, 1024)
        self.res_encoder6 = self.res_block(512, 1024)
        #bottleneck
        self.bottleneck = self.bot_block(1024, 1024)
        self.res_bottleneck = self.res_block(1024, 1024)
        #middle
        self.upsample1 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        #decoder
        self.conv_decoder4 = self.conv_block(1024, 512)
        self.res_decoder4 = self.res_block(1024, 512)
        self.upsample2 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=2)

        self.conv_decoder3 = self.conv_block(512, 256)
        self.res_decoder3 = self.res_block(512, 256)
        self.upsample3 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        self.conv_decoder2 = self.conv_block(256, 128)
        self.res_decoder2 = self.res_block(256, 128)
        self.upsample4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.conv_decoder1 = self.conv_block(128, 64)
        self.res_decoder1 = self.res_block(128, 64)
        self.upsample5 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=2)

        self.conv_decoder0 = self.conv_block(64, 32)
        self.res_decoder0 = self.res_block(64, 32)

        self.final_layer = self.final_block(32, 1)

    def crop(self, x, enc_ftrs):
        chs, dims = x.shape[1:]
        enc_ftrs = torchvision.transforms.CenterCrop([chs, dims])(enc_ftrs)
        return enc_ftrs

    def forward(self, x):

        #encoder
        encode_block1 = self.conv_encoder1(x)
        residual_block1 = self.res_encoder1(x)
        encode_block1 += residual_block1
        encode_block1 = self.maxpool(encode_block1)

        encode_block2 = self.conv_encoder2(encode_block1)
        residual_block2 = self.res_encoder2(encode_block1)
        encode_block2 += residual_block2
        encode_block2 = self.maxpool(encode_block2)

        encode_block3 = self.conv_encoder3(encode_block2)
        residual_block3 = self.res_encoder3(encode_block2)
        encode_block3 += residual_block3
        encode_block3 = self.maxpool(encode_block3)

        encode_block4 = self.conv_encoder4(encode_block3)
        residual_block4 = self.res_encoder4(encode_block3)
        encode_block4 += residual_block4
        encode_block4 = self.maxpool(encode_block4)

        encode_block5 = self.conv_encoder5(encode_block4)
        residual_block5 = self.res_encoder5(encode_block4)
        encode_block5 += residual_block5
        encode_block5 = self.maxpool(encode_block5)

        encode_block6 = self.conv_encoder6(encode_block5)
        residual_block6 = self.res_encoder6(encode_block5)
        encode_block6 += residual_block6
        encode_block6 = self.maxpool(encode_block6)

        #bottleneck
        bottleneck = self.bottleneck(encode_block6)
        residual_bot = self.res_bottleneck(encode_block6)
        bottleneck += residual_bot

        #middle
        decode_middle = self.upsample1(bottleneck)

        #decoder
        cropped5 = self.crop(decode_middle, encode_block5)
        decode_block5 = torch.cat((decode_middle, cropped5), 1)
        cat_layer4 = self.conv_decoder4(decode_block5)
        cat_layer4 += self.res_decoder4(decode_block5)
        cat_layer4 = self.upsample2(cat_layer4)

        cropped4 = self.crop(cat_layer4, encode_block4)
        decode_block4 = torch.cat((cat_layer4, cropped4), 1)
        cat_layer3 = self.conv_decoder3(decode_block4)
        cat_layer3 += self.res_decoder3(decode_block4)
        cat_layer3 = self.upsample3(cat_layer3)

        cropped3 = self.crop(cat_layer3, encode_block3)
        decode_block3 = torch.cat((cat_layer3, cropped3), 1)
        cat_layer2 = self.conv_decoder2(decode_block3)
        cat_layer2 += self.res_decoder2(decode_block3)
        cat_layer2 = self.upsample4(cat_layer2)

        cropped2 = self.crop(cat_layer2, encode_block2)
        decode_block2 = torch.cat((cat_layer2, cropped2), 1)
        cat_layer1 = self.conv_decoder1(decode_block2)
        cat_layer1 += self.res_decoder1(decode_block2)
        cat_layer1 = self.upsample5(cat_layer1)

        cropped1 = self.crop(cat_layer1, encode_block1)
        decode_block1 = torch.cat((cat_layer1, cropped1), 1)
        cat_layer0 = self.conv_decoder0(decode_block1)
        cat_layer0 += self.res_decoder0(decode_block1)

        final_layer = self.final_layer(cat_layer0)

        out = F.interpolate(final_layer, x.shape[2])
        sigmoid = nn.Sigmoid()

        return sigmoid(out)

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
            nn.Dropout(p=p),
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
    def bot_block(in_channels, out_channels, k=5, p=0.5):
        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=k, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=p)
        )
        return block

    @staticmethod
    def final_block(in_channels, out_channels, k=1, p=0.5):

        block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, k, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=p),
        )
        return block