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
        return torch.cat([x1, x2], dim=1)

class Critic(nn.Module):
    def __init__(self, in_channels):
        super(Critic, self).__init__()
        self.conv1 = Conv3DLayer(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = Conv3DLayer(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3_1 = Conv3DLayer(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = Conv3DLayer(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(2)
        
        self.conv4_1 = Conv3DLayer(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = Conv3DLayer(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool3d(2)
        
        self.conv5_1 = Conv3DLayer(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = Conv3DLayer(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = Conv3DLayer(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_4 = Conv3DLayer(256, 1, kernel_size=1, stride=1, padding=0, activation=False)
        
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x

