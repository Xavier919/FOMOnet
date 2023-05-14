import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class FOMOnet(nn.Module):

    def __init__(self, num_channels=1):
        super().__init__()

        #encoder
        self.conv_encoder1 = self._conv_block(num_channels, 32)
        self.conv_encoder2 = self._contraction_block(32, 64)
        self.res_encoder2 = self._residual_block(32, 64)
        self.conv_encoder3 = self._contraction_block(64, 128)
        self.res_encoder3 = self._residual_block(64, 128)
        self.conv_encoder4 = self._contraction_block(128, 256)
        self.res_encoder4 = self._residual_block(128, 256)
        self.conv_encoder5 = self._contraction_block(256, 512)
        self.res_encoder5 = self._residual_block(256, 512)
        self.conv_encoder6 = self._contraction_block(512, 1024)
        self.res_encoder6 = self._residual_block(512, 1024)

        #bottleneck
        self.bottleneck = self._bottleneck_block(1024, 1024)

        #middle
        self.conv_mid_decoder = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)

        #decoder
        self.conv_decoder4 = self._expansion_block(1024, 256)
        self.res_decoder4 = self._residual_block_dec(1024, 256)
        self.transpose4 = nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)

        self.conv_decoder3 = self._expansion_block(512, 128)
        self.res_decoder3 = self._residual_block_dec(512, 128)
        self.transpose3 = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=2, stride=2)

        self.conv_decoder2 = self._expansion_block(256, 64)
        self.res_decoder2 = self._residual_block_dec(256, 64)
        self.transpose2 = nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=2, stride=2)

        self.conv_decoder1 = self._expansion_block(128, 32)
        self.res_decoder1 = self._residual_block_dec(128, 32)
        self.transpose1 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=2, stride=2)

        self.final_layer = self._final_block(64, 32, 1)

    def crop(self, x, enc_ftrs):
        chs, dims = x.shape[1:]
        enc_ftrs = torchvision.transforms.CenterCrop([chs, dims])(enc_ftrs)
        return enc_ftrs

    def forward(self, x):

        #encoder
        encode_block1 = self.conv_encoder1(x)

        encode_block2 = self.conv_encoder2(encode_block1)
        residual_block2 = self.res_encoder2(encode_block1)
        encode_block2 += residual_block2

        encode_block3 = self.conv_encoder3(encode_block2)
        residual_block3 = self.res_encoder3(encode_block2)
        encode_block3 += residual_block3

        encode_block4 = self.conv_encoder4(encode_block3)
        residual_block4 = self.res_encoder4(encode_block3)
        encode_block4 += residual_block4

        encode_block5 = self.conv_encoder5(encode_block4)
        residual_block5 = self.res_encoder5(encode_block4)
        encode_block5 += residual_block5

        encode_block6 = self.conv_encoder6(encode_block5)
        residual_block6 = self.res_encoder6(encode_block5)
        encode_block6 += residual_block6

        #bottleneck
        bottleneck_ = self.bottleneck(encode_block6)

        #middle
        decode_middle = self.conv_mid_decoder(bottleneck_)

        #decoder
        cropped5 = self.crop(decode_middle, encode_block5)
        decode_block5 = torch.cat((decode_middle, cropped5), 1)
        cat_layer4 = self.conv_decoder4(decode_block5)
        cat_layer4 += self.res_decoder4(decode_block5)
        cat_layer4 = self.transpose4(cat_layer4)

        cropped4 = self.crop(cat_layer4, encode_block4)
        decode_block4 = torch.cat((cat_layer4, cropped4), 1)
        cat_layer3 = self.conv_decoder3(decode_block4)
        cat_layer3 += self.res_decoder3(decode_block4)
        cat_layer3 = self.transpose3(cat_layer3)

        cropped3 = self.crop(cat_layer3, encode_block3)
        decode_block3 = torch.cat((cat_layer3, cropped3), 1)
        cat_layer2 = self.conv_decoder2(decode_block3)
        cat_layer2 += self.res_decoder2(decode_block3)
        cat_layer2 = self.transpose2(cat_layer2)

        cropped2 = self.crop(cat_layer2, encode_block2)
        decode_block2 = torch.cat((cat_layer2, cropped2), 1)
        cat_layer1 = self.conv_decoder1(decode_block2)
        cat_layer1 += self.res_decoder1(decode_block2)
        cat_layer1 = self.transpose1(cat_layer1)

        cropped1 = self.crop(cat_layer1, encode_block1)
        decode_block1 = torch.cat((cat_layer1, cropped1), 1)
        final_layer = self.final_layer(decode_block1)

        out = F.interpolate(final_layer, x.shape[2])
        sigmoid = nn.Sigmoid()

        return sigmoid(out)

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size=5):

        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, groups=out_channels, padding='same'),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
        )
        return block

    @staticmethod
    def _contraction_block(in_channels, out_channels, kernel_size=5):

        block = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, groups=out_channels, padding='same'),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
        )
        return block
    
    @staticmethod
    def _residual_block(in_channels, out_channels, kernel_size=5):

        block = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
        )
        return block
    
    @staticmethod
    def _bottleneck_block(in_channels, out_channels, kernel_size=5):
        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=0.2)
        )
        return block

    @staticmethod
    def _expansion_block(in_channels, out_channels, kernel_size=5):

        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, groups=out_channels, padding='same'),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=0.2),
        )
        return block
    
    @staticmethod
    def _residual_block_dec(in_channels, out_channels, kernel_size=5):

        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=0.2),
        )
        return block

    @staticmethod
    def _final_block(in_channels, mid_channels, out_channels, kernel_size=5):

        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(mid_channels),
            nn.Dropout(p=0.2),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size, groups=mid_channels, padding='same'),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(mid_channels),
            nn.Dropout(p=0.2),
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, padding='same'),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
        )
        return block