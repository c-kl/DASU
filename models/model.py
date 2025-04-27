import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from base_networks import *
from JGF_x8 import *
from models.edsr import ResBlock
from torch.autograd import Variable


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    return ret


class DAM(nn.Module):
    def __init__(self, out_channels):
        super(DAM, self).__init__()
    def forward(self, c, x):
        # x: gray image features
        # c: color features
        # l1 distance
        channels = c.shape[1]
        sim_mat_l1 = -torch.abs(x - c)  # <0  (b,c,h,w)
        sim_mat_l1 = torch.sum(sim_mat_l1, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_l1 = torch.sigmoid(sim_mat_l1)  # (0, 0.5) (b,1,h,w)
        sim_mat_l1 = sim_mat_l1.repeat(1, channels, 1, 1)
        sim_mat_l1 = 2 * sim_mat_l1  # (0, 1)

        # cos distance
        sim_mat_cos = x * c  # >0 (b,c,h,w)
        sim_mat_cos = torch.sum(sim_mat_cos, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_cos = torch.tanh(sim_mat_cos)  # (0, 1) (b,1,h,w)
        sim_mat_cos = sim_mat_cos.repeat(1, channels, 1, 1)  # (0, 1)
        sim_mat = sim_mat_l1 * sim_mat_cos  # (0, 1)
        return sim_mat


class fuse(nn.Module):
    def __init__(self, channels):
        super(fuse, self).__init__()
        self.conv1 = nn.Conv2d(channels * 3, channels, 1, padding=0, bias=True)
        self.ca_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )
        self.ca_max = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels*2, channels // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels*3, kernel_size=1),
            nn.Softmax(-1)
        )
    def forward(self, c, x,z):
        r0 = self.ca_avg(c)
        r1 = self.ca_max(c)
        r2 = torch.cat([r0,r1],dim=1)
        r3 = self.conv2(r2)
        G1,G2,G3 = torch.chunk(r3,3,dim=1)
        r4 = G1*c+z
        r5 = G2*c+z
        r6 = G3*c+z
        z0 = r4 + x
        z1 = r5 * x
        z2 = torch.maximum(r6, x)
        z3 = self.conv1(torch.cat([z0, z1, z2], dim=1))
        return z3

class AUF(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv3x3C1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=True),
            nn.PReLU()
        )
        self.conv3x3C2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=True),
            nn.PReLU()
        )
        self.conv1x1 = nn.Conv2d(in_dim * 3, in_dim, kernel_size=1, stride=1, padding=0)
        self.conv3x1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv3x2 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.body1 = ResBlock(conv=default_conv, n_feats=in_dim, kernel_size=3)
        self.body2 = ResBlock(conv=default_conv, n_feats=in_dim, kernel_size=3)
        self.conv3x3 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv3x4 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.alpha1 = nn.Parameter(torch.Tensor([0.5]))
        self.alpha2 = nn.Parameter(torch.Tensor([0.5]))
        self.alpha3 = nn.Parameter(torch.Tensor([1]))
        self.DAM1 = DAM(in_dim)
        self.DAM2 = DAM(in_dim)
        self.fuse1 = fuse(in_dim)
        self.fuse2 = fuse(in_dim)

    def gen_coord(self, in_shape, output_size):
        self.image_size = output_size
        self.coord = make_coord(output_size, flatten=False) \
            .expand(in_shape[0], output_size[0], output_size[1], 2).flip(-1)
        self.coord = self.coord.cuda()

    def forward(self, Target, Guidance, scale, cell):
        Target = F.grid_sample(
            Target, self.coord, mode='bilinear', align_corners=False)
        Guidance = F.grid_sample(
            Guidance, self.coord, mode='bilinear', align_corners=False)
        q_feat1 = self.conv3x3C1(Target)
        guide_hr1 = self.conv3x3C2(Guidance)

        #The frist FEMSF modules
        E1 = self.DAM1(guide_hr1,q_feat1) * guide_hr1
        z1 = self.fuse1(E1, q_feat1,guide_hr1)
        Output1_feature1 = self.conv3x1(z1)

        # The second FEMSF modules
        E2 = self.DAM2(Output1_feature1, q_feat1) * guide_hr1
        z2 = self.fuse2(E2, q_feat1,guide_hr1)
        z3 = self.conv3x2(z2)

        #Output layer
        z4 = self.conv3x4(z3+z1)
        fuse7 = self.body1(z4)
        fuse8 = self.body2(fuse7)
        OUT = self.conv3x3(fuse8 + q_feat1)
        return OUT

#
class Feature_modulatorC(nn.Module):
    def __init__(self, in_dim, out_dim, act=True):
        super().__init__()
    def gen_coord(self, in_shape, output_size):
        self.image_size = output_size
        self.coord = make_coord(output_size, flatten=False) \
            .expand(in_shape[0], output_size[0], output_size[1], 2).flip(-1)
        self.coord = self.coord.cuda()

    def forward(self, feat, cell):
        q_feat = F.grid_sample(
            feat, self.coord, mode='bilinear', align_corners=False)
        # q_feat = self.conv1x1(q_feat)
        return q_feat


class MSRB(nn.Module):
    def __init__(self, outchannel):
        super(MSRB, self).__init__()
        self.conv1_31 = ConvBlock(outchannel, outchannel, kernel_size=7, stride=1, padding=3, activation='prelu',
                                  norm=None)
        self.conv5_51 = ConvBlock(outchannel, outchannel, kernel_size=5, stride=1, padding=2, activation='prelu',
                                  norm=None)
        self.conv1x1u = nn.Conv2d(outchannel * 3, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv1x1d = nn.Conv2d(outchannel * 3, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(outchannel * 2, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv1_32 = ConvBlock(outchannel, outchannel, kernel_size=5, stride=1, padding=2,
                                  activation='prelu', norm=None)
        self.conv5_52 = ConvBlock(outchannel, outchannel, kernel_size=3, stride=1, padding=1,
                                  activation='prelu', norm=None)
        self.body1 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.body2 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.conv3x3 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.conv1_31(x)
        x1 = self.conv5_51(x)
        fuse1 = x0 * x1
        fuse2 = x0 + x1
        fuse3 = torch.maximum(x0, x1)
        fuse4 = torch.cat([fuse1, fuse2, fuse3], dim=1)
        x2 = self.conv1x1u(fuse4)
        x3 = self.conv1_32(x2)
        x4 = self.conv1x1d(fuse4)
        x5 = self.conv5_52(x4)
        x6 = torch.cat([x3, x5], dim=1)
        x7 = self.conv1x1(x6)
        x8 = self.body1(x7)
        x9 = self.body2(x8)
        out = self.conv3x3(x9 + x)
        return out

class LNet(nn.Module):
    def __init__(self, outchannel):
        super(LNet, self).__init__()
        self.outchannel = outchannel
        self.CRB1 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB2 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB3 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB4 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.conv1 = nn.Conv2d(outchannel * 2, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv_k2 = nn.Conv2d(outchannel * 2, outchannel, kernel_size=1, stride=1, padding=0)

        self.bata1 = nn.Parameter(torch.Tensor([0.5]))
        self.bata2 = nn.Parameter(torch.Tensor([0.5]))
        self.ita = nn.Parameter(torch.Tensor([0.5]))
        self.delta1 = nn.Parameter(torch.Tensor([0.5]))
        self.avg_poolA = nn.AdaptiveAvgPool2d(1)
        self.avg_poolM = nn.AdaptiveMaxPool2d(1)

        self.downsample = Feature_modulatorC(16, 16)
        self.upsample = Feature_modulatorC(16, 16)
        self.MSRB = MSRB(outchannel)

        self.LF_kenel_conv2d_3x3 = nn.Conv2d(outchannel,outchannel,3,1,1)
        self.LF_kenel_conv2d_5x5 = nn.Conv2d(outchannel, outchannel, 5, 1, 2)
    def down_coord(self, coord, in_size, out_size, m_scal_factor, N):
        down_h = int(in_size[0] * m_scal_factor[0])
        down_w = int(in_size[1] * m_scal_factor[1])
        self.downsample.gen_coord((N,\
            1, in_size[0], in_size[1]), (down_h, down_w))

    def up_coord(self, coord, in_size, out_size, m_scal_factor, N):
            up_h = int(out_size[0] * (m_scal_factor[0]))
            up_w = int(out_size[1] * (m_scal_factor[1]))
            self.upsample.gen_coord((N, \
                                     1, out_size[0], out_size[1]), (up_h, up_w))
    def forward(self, LF_up_In, D_up_In,L0,HF_up_In, scale,coord):
        N, in_h, in_w = LF_up_In.shape[0], LF_up_In.shape[-2], LF_up_In.shape[-1]
        out_h = L0.shape[-2]
        out_w = L0.shape[-1]
        down_size = self.down_coord(coord, (out_h,out_w), (out_h, out_w), (1,1), N)
        up_size = self.up_coord(coord, (in_h, in_w), (in_h, in_w), (1, 1), N)
        x0 = self.CRB1(LF_up_In) #In paper, LF_up_In is L_k-1
        x1 = self.downsample(x0, True)
        x2 = self.CRB2(x1)
        x3 = x2 - L0
        x4 = self.CRB3(x3)
        x5 = self.upsample(x4, True)
        x6 = self.CRB4(x5)
        x7 = x6 * self.bata1
        k2_3x3 = self.LF_kenel_conv2d_3x3(D_up_In)#In paper, D_up_In is D_k-1
        k2_5x5 = self.LF_kenel_conv2d_5x5(D_up_In)
        k2 = self.conv_k2(torch.cat([k2_3x3,k2_5x5],dim=1))
        k3 = self.bata2* (LF_up_In - k2)
        cc = self.ita*(D_up_In - LF_up_In - HF_up_In)#In paper, HF_up_In is H_k-1
        x8 = LF_up_In - (x7 + k3 + cc) * self.delta1
        Lk = self.MSRB(x8)
        return Lk
class HNet(nn.Module):
    def __init__(self, outchannel):
        super(HNet, self).__init__()
        self.outchannel = outchannel
        self.CRB1 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB2 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB3 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB4 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.conv3 = nn.Conv2d(outchannel * 2, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv_k2 = nn.Conv2d(outchannel * 2, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv_k4 = nn.Conv2d(outchannel * 2, outchannel, kernel_size=1, stride=1, padding=0)

        self.gamma1 = nn.Parameter(torch.Tensor([0.5]))
        self.gamma2 = nn.Parameter(torch.Tensor([0.5]))
        self.delta2 = nn.Parameter(torch.Tensor([0.5]))
        self.ita = nn.Parameter(torch.Tensor([0.5]))
        self.avg_poolA = nn.AdaptiveAvgPool2d(1)
        self.avg_poolM = nn.AdaptiveMaxPool2d(1)

        self.downsample = Feature_modulatorC(16, 16)
        self.upsample = Feature_modulatorC(16, 16)
        self.MSRB = MSRB(outchannel)
        self.HF_kenel_conv2d_3x3 = nn.Conv2d(outchannel,outchannel,3,1,1)
        self.HF_kenel_conv2d_5x5 = nn.Conv2d(outchannel, outchannel, 5, 1, 2)
    def down_coord(self, coord, in_size, out_size, m_scal_factor, N):
        down_h = int(in_size[0] * m_scal_factor[0])
        down_w = int(in_size[1] * m_scal_factor[1])
        self.downsample.gen_coord((N,\
            1, in_size[0], in_size[1]), (down_h, down_w))
    def up_coord(self, coord, in_size, out_size, m_scal_factor, N):
            up_h = int(out_size[0] * (m_scal_factor[0]))
            up_w = int(out_size[1] * (m_scal_factor[1]))
            self.upsample.gen_coord((N, \
                                     1, out_size[0], out_size[1]), (up_h, up_w))
    def forward(self, HF_up_In, D_up_In, H0,Lk, scale,coord):
        N, in_h, in_w = HF_up_In.shape[0], HF_up_In.shape[-2], HF_up_In.shape[-1]
        out_h = H0.shape[-2]
        out_w = H0.shape[-1]
        down_size = self.down_coord(coord, (out_h, out_w), (out_h, out_w), (1, 1), N)
        up_size = self.up_coord(coord, (in_h, in_w), (in_h, in_w), (1, 1), N)
        x0 = self.CRB1(HF_up_In) #In paper, HF_up_In is H_k-1
        x1 = self.downsample(x0, True)
        x2 = self.CRB2(x1)
        x3 = x2 - H0
        x4 = self.CRB3(x3)
        x5 = self.upsample(x4, True)
        x6 = self.CRB4(x5)
        x7 = x6 * self.gamma1
        k2_3x3 = self.HF_kenel_conv2d_3x3(D_up_In) #In paper, D_up_In is D_k-1
        k2_5x5 = self.HF_kenel_conv2d_5x5(D_up_In)
        k2 = self.conv_k2(torch.cat([k2_3x3,k2_5x5],dim=1))
        k3 = (HF_up_In - k2)* self.gamma2
        cc = self.ita * (D_up_In - Lk - HF_up_In)
        x8 = HF_up_In - (x7 + k3 + cc) * self.delta2
        x11 = self.MSRB(x8)
        return x11

class DNet(nn.Module):
    def __init__(self, outchannel):
        super(DNet, self).__init__()
        self.outchannel = outchannel
        self.CRB1 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB2 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB3 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB4 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)

        self.gamma2 = nn.Parameter(torch.Tensor([0.5]))
        self.bata2 = nn.Parameter(torch.Tensor([0.5]))
        self.delta3 = nn.Parameter(torch.Tensor([0.5]))
        self.ita = nn.Parameter(torch.Tensor([0.5]))
        self.avg_poolA1 = nn.AdaptiveAvgPool2d(1)
        self.avg_poolM1 = nn.AdaptiveMaxPool2d(1)

        self.downsample = Feature_modulatorC(16, 16)
        self.upsample = Feature_modulatorC(16, 16)
        self.MSRB = MSRB(outchannel)
        self.conv_k2 = nn.Conv2d(outchannel * 2, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv_k4 = nn.Conv2d(outchannel * 2, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv_kk2 = nn.Conv2d(outchannel * 2, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv_kk4 = nn.Conv2d(outchannel * 2, outchannel, kernel_size=1, stride=1, padding=0)

        self.HF_kenel_conv2d_3x3 = nn.Conv2d(outchannel, outchannel, 3, 1, 1)
        self.HF_kenel_conv2d_5x5 = nn.Conv2d(outchannel, outchannel, 5, 1, 2)
        self.LF_kenel_conv2d_3x3 = nn.Conv2d(outchannel, outchannel, 3, 1, 1)
        self.LF_kenel_conv2d_5x5 = nn.Conv2d(outchannel, outchannel, 5, 1, 2)

    def down_coord(self, coord, in_size, out_size, m_scal_factor, N):
        down_h = int(in_size[0] * m_scal_factor[0])
        down_w = int(in_size[1] * m_scal_factor[1])
        self.downsample.gen_coord((N, \
                                   1, in_size[0], in_size[1]), (down_h, down_w))

    def up_coord(self, coord, in_size, out_size, m_scal_factor, N):
        up_h = int(out_size[0] * (m_scal_factor[0]))
        up_w = int(out_size[1] * (m_scal_factor[1]))
        self.upsample.gen_coord((N, \
                                 1, out_size[0], out_size[1]), (up_h, up_w))
    def forward(self, D_up_In, Lk, Hk, D0, scale, coord):
        N, in_h, in_w = D_up_In.shape[0], D_up_In.shape[-2], D_up_In.shape[-1]
        out_h = D0.shape[-2]
        out_w = D0.shape[-1]
        down_size = self.down_coord(coord, (out_h, out_w), (out_h, out_w), (1, 1), N)
        up_size = self.up_coord(coord, (in_h, in_w), (in_h, in_w), (1, 1), N)

        x0 = self.CRB1(D_up_In) #In paper, D_up_In is D_K-1
        x1 = self.downsample(x0, True)
        x2 = self.CRB2(x1)
        x3 = x2 - D0
        x4 = self.CRB3(x3)
        x5 = self.upsample(x4, True)
        x6 = self.CRB4(x5)

        k2_3x3 = self.HF_kenel_conv2d_3x3(D_up_In)
        k2_5x5 = self.HF_kenel_conv2d_5x5(D_up_In)
        k2 = self.conv_k2(torch.cat([k2_3x3, k2_5x5], dim=1))
        k3 = Hk - k2

        invers_weight_HF_3x3 = self.HF_kenel_conv2d_3x3.weight.data.flip(dims=[2]).flip(dims=[3])
        invers_weight_HF_3x3 = nn.Parameter(data=invers_weight_HF_3x3, requires_grad=False)
        k4_3x3 = F.conv2d(k3, invers_weight_HF_3x3, padding=1)

        invers_weight_HF_5x5 = self.HF_kenel_conv2d_5x5.weight.data.flip(dims=[2]).flip(dims=[3])
        invers_weight_HF_5x5 = nn.Parameter(data=invers_weight_HF_5x5, requires_grad=False)
        k4_5x5 = F.conv2d(k3, invers_weight_HF_5x5, padding=2)
        k4 = self.conv_k4(torch.cat([k4_3x3, k4_5x5], dim=1)) * self.bata2

        kk2_3x3 = self.LF_kenel_conv2d_3x3(D_up_In)
        kk2_5x5 = self.LF_kenel_conv2d_5x5(D_up_In)
        kk2 = self.conv_kk2(torch.cat([kk2_3x3, kk2_5x5], dim=1))
        kk3 = Lk - kk2
        invers_weight_LF_3x3 = self.LF_kenel_conv2d_3x3.weight.data.flip(dims=[2]).flip(dims=[3])
        invers_weight_LF_3x3 = nn.Parameter(data=invers_weight_LF_3x3, requires_grad=False)
        kk4_3x3 = F.conv2d(kk3, invers_weight_LF_3x3, padding=1)

        invers_weight_LF_5x5 = self.LF_kenel_conv2d_5x5.weight.data.flip(dims=[2]).flip(dims=[3])
        invers_weight_LF_5x5 = nn.Parameter(data=invers_weight_LF_5x5, requires_grad=False)
        kk4_5x5 = F.conv2d(kk3, invers_weight_LF_5x5, padding=2)
        kk4 = self.conv_kk4(torch.cat([kk4_3x3, kk4_5x5], dim=1)) * self.gamma2
        cc = self.ita * (D_up_In - Lk - Hk)

        x8 = D_up_In - (x6 + k4 + kk4 + cc) * self.delta3
        DK = self.MSRB(x8)
        return DK


class DASUNet(nn.Module):
    def __init__(self, args, outchannel=26):
        super().__init__()
        self.args = args
        self.LNet1 = LNet(outchannel)
        self.HNet1 = HNet(outchannel)
        self.DNet1 = DNet(outchannel)

        self.LNet2 = LNet(outchannel)
        self.HNet2 = HNet(outchannel)
        self.DNet2 = DNet(outchannel)

        self.upsample1D = AUF(outchannel)
        self.upsample1HF = AUF(outchannel)
        self.upsample1LF = AUF(outchannel)

        self.upsample2D = AUF(outchannel)
        self.upsample2HF = AUF(outchannel)
        self.upsample2LF = AUF(outchannel)

        self.conv1 = nn.Conv2d(3, outchannel, kernel_size=1, stride=1, padding=0)
        self.CRB1 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB11 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.conv2 = nn.Conv2d(1, outchannel, kernel_size=1, stride=1, padding=0)
        self.CRB2 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB22 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.conv3 = nn.Conv2d(1, outchannel, kernel_size=1, stride=1, padding=0)
        self.CRB3 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB33 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.conv4 = nn.Conv2d(1, outchannel, kernel_size=1, stride=1, padding=0)
        self.CRB4 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.CRB44 = ResBlock(conv=default_conv, n_feats=outchannel, kernel_size=3)
        self.convOUT = nn.Conv2d(outchannel, 1, 3, 1, 1)
        self.convOUT_HF = nn.Conv2d(outchannel, 1, 3, 1, 1)
        self.convOUT_LF = nn.Conv2d(outchannel, 1, 3, 1, 1)

    def gen_DSF_coord(self, coord, in_size, out_size, m_scal_factor, N):
        m_size_h = int(in_size[0] * m_scal_factor[0])
        m_size_w = int(in_size[1] * m_scal_factor[1])

        self.upsample1D.gen_coord((N, \
                                   1, in_size[0], in_size[1]), (m_size_h, m_size_w))
        self.upsample1HF.gen_coord((N, \
                                   1, in_size[0], in_size[1]), (m_size_h, m_size_w))
        self.upsample1LF.gen_coord((N, \
                                   1, in_size[0], in_size[1]), (m_size_h, m_size_w))

        self.upsample2D.coord = coord.flip(-1)
        self.upsample2HF.coord = coord.flip(-1)
        self.upsample2LF.coord = coord.flip(-1)
        return [m_size_h, m_size_w]
    def forward(self, data):
        color, lr_depth, lr_depth_up, depth_HF_lr, depth_HF_lr_up, depth_LF_lr, \
        depth_LF_lr_up, coord, field = data['hr_image'], data['lr_depth'], data['lr_depth_up'], \
                                       data['depth_HF_lr'], data['depth_HF_lr_up'], data['depth_LF_lr'], \
                                       data['depth_LF_lr_up'], data['hr_coord'], data["field"]
        #color feature initialization
        C= self.conv1(color)
        C = self.CRB1(C)
        C = self.CRB11(C)

        # depth feature initialization
        Dl = self.conv2(lr_depth)
        D = self.CRB2(Dl)
        D0 = self.CRB22(D)

        # LF feature initialization
        Ll = depth_LF_lr
        LF = self.conv3(Ll)
        LF = self.CRB3(LF)
        L0 = self.CRB33(LF)

        # HF feature initialization
        Hl = depth_HF_lr
        HF = self.conv4(Hl)
        HF = self.CRB4(HF)
        H0 = self.CRB44(HF)

        # arbitrary-scale Up-down sample function
        scale = field.mean(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        N, in_h, in_w = D0.shape[0], D0.shape[-2], D0.shape[-1]
        out_h = int(coord.shape[1])
        out_w = int(coord.shape[2])
        sum_scal_factor_h = out_h / in_h
        sum_scal_factor_w = out_w / in_w
        scal_factor_h = sum_scal_factor_h ** 0.5
        scal_factor_w = sum_scal_factor_w ** 0.5
        m_size = self.gen_DSF_coord(coord, (in_h, in_w), (out_h, out_w), (scal_factor_h, scal_factor_w), N)

        #Using the AUF module to perform dual-mode fusion and intermediate resolution up-sample about depth, HF and LF feature
        D_up_In = self.upsample1D(D0, C, scale, True)# In paper, D_up_In is D0
        HF_up_In = self.upsample1HF(H0, C, scale, True)# In paper, HF_up_In is H0
        LF_up_In = self.upsample1LF(L0, C, scale, True)# In paper, LF_up_In is L0

        #Stage 1
        L1 = self.LNet1(LF_up_In, D_up_In, L0, HF_up_In, scale, coord)
        H1 = self.HNet1(HF_up_In, D_up_In, H0, L1, scale, coord)
        D1 = self.DNet1(D_up_In, L1, H1, D0, scale, coord)

        #Using the AUF module to perform dual-mode fusion and target resolution up-sample about depth, HF and LF feature
        D_up_Ta = self.upsample2D(D1, C, scale, True)
        HF_up_Ta = self.upsample2HF(H1, C, scale, True)
        LF_up_Ta = self.upsample2LF(L1, C, scale, True)

        # Stage 2
        L2 = self.LNet2(LF_up_Ta, D_up_Ta, L1, HF_up_Ta, scale, coord)
        H2 = self.HNet2(HF_up_Ta, D_up_Ta, H1, L2, scale, coord)
        D2 = self.DNet2(D_up_Ta, L2, H2, D1, scale, coord)

        #Output layer
        DSR = self.convOUT(D2) + lr_depth_up
        LSR = self.convOUT_LF(L2) + depth_LF_lr_up
        HSR = self.convOUT_HF(H2) + depth_HF_lr_up
        return DSR, LSR, HSR