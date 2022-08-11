import mmcv
import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from mmflow.datasets import visualize_flow
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.arch_util import ResidualBlockNoBN, flow_warp, make_layer


class BasicVSR(nn.Module):
    """BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond

    Only support x4 upsampling.

    Args:
        num_feat (int): Channel number of intermediate features. 中间层特征的channel
            Default: 64.
        num_block (int): Block number of residual blocks in each propagation branch. 每一个传播分支 的残差模块
            Default: 30.
        spynet_path (str): The path of Pre-trained SPyNet model. # SPynet 预训练模型一部分
            Default: None.
    """

    def __init__(self, num_feat=64, num_block=30, spynet_path=None):
        super(BasicVSR, self).__init__()
        self.num_feat = num_feat
        self.num_block = num_block

        # Flow-based Feature Alignment
        self.spynet = SpyNet(load_path=spynet_path)  # component 1: 初始化 SpyNet 光流模型，是对齐Alignment 组件

        # Bidirectional Propagation # component 2： 双向 propagation
        self.forward_resblocks = ConvResBlock(num_feat + 3, num_feat, num_block)
        self.backward_resblocks = ConvResBlock(num_feat + 3, num_feat, num_block)

        # Concatenate Aggregation # component 3: concate 聚合 Aggregation
        self.concate = nn.Conv2d(num_feat * 2, num_feat, kernel_size=1, stride=1, padding=0, bias=True)

        # Pixel-Shuffle Upsampling # component 4: 上采样 upsampling
        self.up1 = PSUpsample(num_feat, num_feat, scale_factor=2)
        self.up2 = PSUpsample(num_feat, 64, scale_factor=2)

        # The channel of the tail layers is 64 # 末尾 feature map channel 是64
        self.conv_hr = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # Global Residual Learning # 全局 残差学习
        self.img_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # Activation Function # 激活层 leakyRelu
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def comp_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping. 运动补偿就是为了 帧对齐

        Args:
            lrs (tensor): LR frames, the shape is (n, t, c, h, w) low resolution 低帧

        Return:
            tuple(Tensor): Optical flow. 返回一个光流 tensor形式
            forward_flow refers to the flow from current frame to the previous frame. 顺流
            backward_flow is the flow from current frame to the next frame . 逆流
        """
        # [num, time, channel, height, width]
        n, t, c, h, w = lrs.size()
        # lrs[:, 1:, :, :, :].shape = lrs[3, 3, 3, 64, 64].reshape(-1, 3, 64, 64) = [9, 3, 64, 64]
        forward_lrs = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)  # n t c h w -> (n t) c h w -->1指的是t=1， 表示的是当前帧（t=0）的后一帧
        # lrs[:, :-1, :, :, :].shape = lrs[3, 3, 3, 64, 64].reshape(-1, 3, 64, 64) = [9, 3, 64, 64]
        backward_lrs = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)  # n t c h w -> (n t) c h w -->-1指的是t=-1, 表示的是当前帧的前一帧

        forward_flow = self.spynet(forward_lrs, backward_lrs).view(n, t - 1, 2, h, w)  # 让前一帧和后一帧做对齐操作
        backward_flow = self.spynet(backward_lrs, forward_lrs).view(n, t - 1, 2, h, w)  # [3, 3, 2, 64, 64]
        """
        test
        """
        flow = forward_flow.reshape(2, 192, 192).permute(1, 2, 0)
        sparse_flow(flow)
        # ba_flow = backward_flow.reshape(2, 192, 192).permute(1, 2, 0)
        # sparse_flow(ba_flow)
        return forward_flow, backward_flow

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()

        assert h >= 64 and w >= 64, (
            'The height and width of input should be at least 64, '
            f'but got {h} and {w}.')

        forward_flow, backward_flow = self.comp_flow(lrs)

        # forward_flow = rearrange(forward_flow, 'n t c h w -> t n h w c').contiguous()
        # backward_flow = rearrange(backward_flow, 'n t c h w -> t n h w c').contiguous()
        # lrs = rearrange(lrs, 'n t c h w -> t n c h w').contiguous()

        # Backward Propagation
        rlt = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:  # 就是最后一个t 不执行if语句
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([curr_lr, feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            rlt.append(feat_prop)
        rlt = rlt[::-1]

        # Forward Propagation
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([curr_lr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # Fusion and Upsampling
            cat_feat = torch.cat([rlt[i], feat_prop], dim=1)
            sr_rlt = self.lrelu(self.concate(cat_feat))
            sr_rlt = self.lrelu(self.up1(sr_rlt))
            sr_rlt = self.lrelu(self.up2(sr_rlt))
            sr_rlt = self.lrelu(self.conv_hr(sr_rlt))
            sr_rlt = self.conv_last(sr_rlt)

            # Global Residual Learning
            base = self.img_up(curr_lr)

            sr_rlt += base
            rlt[i] = sr_rlt

        return torch.stack(rlt, dim=1)


#############################
# Conv + ResBlock
class ConvResBlock(nn.Module):
    def __init__(self, in_feat, out_feat=64, num_block=30):
        super(ConvResBlock, self).__init__()

        conv_resblock = []
        conv_resblock.append(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True))
        conv_resblock.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        conv_resblock.append(make_layer(ResidualBlockNoBN, num_block, num_feat=out_feat))

        self.conv_resblock = nn.Sequential(*conv_resblock)

    def forward(self, x):
        return self.conv_resblock(x)


#############################
# Upsampling with Pixel-Shuffle
class PSUpsample(nn.Module):
    def __init__(self, in_feat, out_feat, scale_factor):
        super(PSUpsample, self).__init__()

        self.scale_factor = scale_factor
        self.up_conv = nn.Conv2d(in_feat, out_feat * scale_factor * scale_factor, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.up_conv(x)
        return F.pixel_shuffle(x, upscale_factor=self.scale_factor)


def sparse_flow(flow):
    flow = flow.detach().numpy()
    visualize_flow(flow, save_file='flow_map.png')


if __name__ == '__main__':
    model = BasicVSR()
    lrs = torch.randn(3, 4, 3, 64, 64)
    rlt = model(lrs)

    print(rlt.size())
