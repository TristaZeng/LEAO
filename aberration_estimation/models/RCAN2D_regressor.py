import torch
import torch.nn as nn
from models import register
import torch.nn.functional as F
import math


class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=[1,1])
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0)
    def forward(self,input):
        W = self.avgpool(input)
        W = self.conv1(W)
        W = F.relu(W)
        W = self.conv2(W)
        W = F.sigmoid(W)
        mul = torch.mul(input, W)
        return mul

class RCAB(nn.Module):
    def __init__(self,channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.CALayer = CALayer(channel)
    def forward(self,input):
        conv = self.conv1(input)
        conv = F.leaky_relu(conv,negative_slope=0.2)
        conv = self.conv2(conv)
        conv = F.leaky_relu(conv,negative_slope=0.2)
        att = self.CALayer(conv)
        output = torch.add(att, input)
        return output

class ResidualGroup(nn.Module):
    def __init__(self,channel=64,n_RCAB=5,sampling_op=None):
        super().__init__()
        self.RCAB = []
        for _ in range(n_RCAB):
            self.RCAB.append(RCAB(channel))
        if sampling_op is not None:
            self.RCAB.append(sampling_op)
        self.RCAB = nn.Sequential(*self.RCAB)
    def forward(self,input):       
        return self.RCAB(input)

class RCAN2D(nn.Module):
    def __init__(self,inp_angle=49,reg_channel=64,channel=64,n_ResGroup=3,n_RCAB=2):
        super().__init__()

        self.channel = channel
        self.conv1 = nn.Conv2d(inp_angle,channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel,channel, kernel_size=3, padding=1)
        self.ResidualGroup_enc = []
        for _ in range(n_ResGroup):
            self.ResidualGroup_enc.append(ResidualGroup(channel,n_RCAB=n_RCAB,sampling_op=nn.MaxPool2d(kernel_size=2)))
        self.ResidualGroup_enc = nn.Sequential(*self.ResidualGroup_enc)
        
        # branching to decoder
        self.ResidualGroup_dec = []
        for _ in range(n_ResGroup):
            self.ResidualGroup_dec.append(ResidualGroup(channel,n_RCAB=n_RCAB,sampling_op=nn.UpsamplingBilinear2d(scale_factor=2)))
        self.ResidualGroup_dec = nn.Sequential(*self.ResidualGroup_dec)
        self.conv1_1 = nn.Conv2d(channel,256, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(256,inp_angle, kernel_size=3, padding=1)
        
        # branching to regressor
        self.conv2_1 = nn.Sequential(nn.Conv2d(channel//2, reg_channel, kernel_size=3, padding=1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(reg_channel, reg_channel, kernel_size=3, padding=1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(reg_channel, reg_channel, kernel_size=3, padding=1),
                                nn.LeakyReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=[1,1]) # avgpool returns [bs,ch,1,1,1]
        self.fc = nn.Linear(reg_channel, 17)
        
    def forward(self,inputs):
        conv = self.conv1(inputs)
        conv = F.relu(conv)
        conv = self.conv2(conv)
        conv = F.relu(conv)
        conv = self.ResidualGroup_enc(conv)
        latent = conv
        
        # decoder
        conv = self.ResidualGroup_dec(conv)
        conv = self.conv1_1(conv)
        conv = F.leaky_relu(conv,negative_slope=0.2)
        conv = self.conv1_2(conv)
        conv = F.leaky_relu(conv,negative_slope=0.2)
        
        # estimator
        zern_coeff = self.conv2_1(latent[:,0:self.channel//2,:,:])
        zern_coeff = self.avgpool(zern_coeff)
        zern_coeff = zern_coeff.view(zern_coeff.size(0), -1) # reshape zern_coeff to [bs,-1]   
        zern_coeff = self.fc(zern_coeff)
        
        return latent,zern_coeff,conv
    
@register('rcan_regressor2d_v7')
def make_rcan_regressor2d(inp_angle,reg_channel,channel,n_ResGroup,n_RCAB):
    return RCAN2D(inp_angle=inp_angle,reg_channel=reg_channel,channel=channel,n_ResGroup=n_ResGroup,n_RCAB=n_RCAB)
