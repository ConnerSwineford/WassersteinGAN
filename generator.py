import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, activation=True):
        super(Conv3DLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.relu(x)
        return x

class Conv3DLayerBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Conv3DLayerBN, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Deconv3DLayerBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1):
        super(Deconv3DLayerBN, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CropAndConcatLayer(nn.Module):
    def forward(self, x1, x2):
        diff_depth = x2.size(2) - x1.size(2)
        diff_height = x2.size(3) - x1.size(3)
        diff_width = x2.size(4) - x1.size(4)

        x2 = x2[:, :, diff_depth // 2 : x2.size(2) - diff_depth // 2,
                 diff_height // 2 : x2.size(3) - diff_height // 2,
                 diff_width // 2 : x2.size(4) - diff_width // 2]
        
        if x2.size(2) != x1.size(2):
            x2 = x2[:, :, :-1, :, :]
        if x2.size(3) != x1.size(3):
            x2 = x2[:, :, :, :-1, :]
        if x2.size(4) != x1.size(4):
            x2 = x2[:, :, :, :, :-1]
        
        return torch.cat([x1, x2], dim=1)

class Generator(nn.Module):
    def __init__(self, in_channels):
        super(Generator, self).__init__()

        self.conv1_1 = Conv3DLayerBN(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = Conv3DLayerBN(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2_1 = Conv3DLayerBN(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = Conv3DLayerBN(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3_1 = Conv3DLayerBN(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = Conv3DLayerBN(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4_1 = Conv3DLayerBN(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = Conv3DLayerBN(128, 128, kernel_size=3, stride=1, padding=1)

        self.upconv3 = Deconv3DLayerBN(128, 64, kernel_size=4, stride=2, padding=1)
        self.concat3 = CropAndConcatLayer()
        self.conv5_1 = Conv3DLayerBN(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = Conv3DLayerBN(64, 64, kernel_size=3, stride=1, padding=1)

        self.upconv2 = Deconv3DLayerBN(64, 32, kernel_size=4, stride=2, padding=1)
        self.concat2 = CropAndConcatLayer()
        self.conv6_1 = Conv3DLayerBN(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = Conv3DLayerBN(32, 32, kernel_size=3, stride=1, padding=1)

        self.upconv1 = Deconv3DLayerBN(32, 16, kernel_size=4, stride=2, padding=1)
        self.concat1 = CropAndConcatLayer()
        self.conv7_1 = Conv3DLayerBN(32, 16, kernel_size=3, stride=1, padding=1)

        self.output_conv = Conv3DLayer(16, 1, kernel_size=3, stride=1, padding=1, activation=False)

    def forward(self, x):

        input_shape = x.shape

        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4_1(pool3)
        conv4 = self.conv4_2(conv4)

        upconv3 = self.upconv3(conv4)
        concat3 = self.concat3(upconv3, conv3)
        conv5 = self.conv5_1(concat3)
        conv5 = self.conv5_2(conv5)

        upconv2 = self.upconv2(conv5)
        concat2 = self.concat2(upconv2, conv2)
        conv6 = self.conv6_1(concat2)
        conv6 = self.conv6_2(conv6)

        upconv1 = self.upconv1(conv6)
        concat1 = self.concat1(upconv1, conv1)
        conv7 = self.conv7_1(concat1)

        outputs = self.output_conv(conv7)

        pad_depth = input_shape[2] - outputs.shape[2]
        pad_height = input_shape[3] - outputs.shape[3]
        pad_width = input_shape[4] - outputs.shape[4]

        outputs = F.pad(outputs, (pad_width // 2, pad_width - pad_width // 2,
                                  pad_height // 2, pad_height - pad_height // 2,
                                  pad_depth // 2, pad_depth - pad_depth // 2))
        
        print(outputs.shape)

        return outputs

