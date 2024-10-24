import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class FreqAttn(nn.Module):
    def __init__(self, in_channels, act='sigmoid', spatial_group=1, spatial_kernel=3, init='zero'):
        super().__init__()
        self.in_channels = in_channels
        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group
        self.attn_conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.act = act

        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.wti = DWTInverse(mode='zero', wave='haar')

    def compute_frequency_weights(self, tensors):
        yL, yH = self.wt(tensors)
        yHn = torch.sum(yH[0], dim=2)
        low_freq_img = F.interpolate(yL, scale_factor=2., mode='bilinear')
        high_freq_img = F.interpolate(yHn, scale_factor=2., mode='bilinear')

        return low_freq_img, high_freq_img

    def forward(self, x, y):

        b, _, h, w = x.shape

        low_freq_img, high_freq_img = self.compute_frequency_weights(x)

        attn = self.attn_conv(torch.cat([low_freq_img, high_freq_img], dim=1)).sigmoid()
        out = y[0] * attn[:, 0, :, :].unsqueeze(1) + y[1] * attn[:, 1, :, :].unsqueeze(1) + y[2] * attn[:, 2, :, :].unsqueeze(1)

        return out


class MSKblock(nn.Module):
    def __init__(self, dim, residual=False):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv3 = nn.Conv2d(dim, dim // 2, 1)
        self.conv4 = nn.Conv2d(dim, dim // 2, 1)
        self.conv5 = nn.Conv2d(dim, dim // 2, 1)
        self.freq_conv = nn.Conv2d(dim, dim // 2, 1)
        self.freq_processor = FreqAttn(dim // 2)
        self.conv_squeeze = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=7, padding=3)
        self.conv = nn.Conv2d(dim, dim, 1)
        self.residual = residual

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv1(attn1)
        attn3 = self.conv2(attn2)

        attn1 = self.conv3(attn1)
        attn2 = self.conv4(attn2)
        attn3 = self.conv5(attn3)

        attn = torch.cat([attn1, attn2, attn3], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        conv_attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1) + attn3 * sig[:, 2, :,
                                                                                                          :].unsqueeze(
            1)
        freq_attn = self.freq_processor(self.freq_conv(x), [attn1, attn2, attn3])
        attn = torch.cat([conv_attn, freq_attn], dim=1)
        attn = self.conv(attn)
        out = x * attn
        if self.residual:
            out += x
        return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, d_model, residual=False):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = MSKblock(d_model, residual)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, residual=False):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, residual)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class arfs_block(nn.Module):
    def __init__(self, dims, drop=0., drop_path=0., act_layer=nn.GELU, residual=False):
        super().__init__()

        self.block1 = Block(dims[0], True)
        self.block2 = Block(dims[1], True)
        self.block3 = Block(dims[2], True)

        self.down_conv1 = nn.Conv2d(dims[0], dims[1], 3, 2, 1)
        self.down_conv2 = nn.Conv2d(dims[1], dims[2], 3, 2, 1)

    def forward(self, features):  # B C H W
        x1, x2, x3 = features

        x1_1 = self.block1(x1)
        y1 = x1_1 + x1
        x1_1 = self.down_conv1(x1_1)

        x2_2 = self.block2(x2 + x1_1)
        y2 = x2_2 + x2
        x2_2 = self.down_conv2(x2_2)

        x3_3 = self.block3(x3 + x2_2)
        y3 = x3_3 + x3

        return [y1, y2, y3]


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    device = torch.device("cpu")
    module = arfs_block(dims=[256, 256, 256]).to(device)
    params = sum(p.numel() for p in module.parameters())
    print(params)
    x1 = torch.randn(2, 256, 80, 80).to(device)
    x2 = torch.randn(2, 256, 40, 40).to(device)
    x3 = torch.randn(2, 256, 20, 20).to(device)
    out = module([x1, x2, x3])
    for x in out:
        print(x.size())
