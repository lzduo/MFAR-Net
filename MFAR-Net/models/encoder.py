import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.get_activations import get_activation
from .ARFS import arfs_block


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(ch_in,
                              ch_out,
                              kernel_size,
                              stride,
                              padding=(kernel_size - 1) // 2 if padding is None else padding,
                              bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class CSMHCALayer(nn.Module):     # Cross Scale Multi-Head Cross Attention
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src_h, src_l, src_mask=None, pos_embed_h=None, pos_embed_l=None) -> torch.Tensor:
        if self.normalize_before:
            src_h = self.norm1(src_h)
            src_l = self.norm1(src_l)
        q = self.with_pos_embed(src_h, pos_embed_h)
        k = self.with_pos_embed(src_l, pos_embed_l)
        src, _ = self.self_attn(q, k, value=src_l, attn_mask=src_mask)

        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class MFIA(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward, dropout, activation):
        super().__init__()
        self.mhsa_layer = TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.csmhca_layer1 = CSMHCALayer(hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.csmhca_layer2 = CSMHCALayer(hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1)

    def forward(self, src_h_flatten, src_m_flatten, src_l_flatten, h, w, pos_embed_h=None, pos_embed_m=None, pos_embed_l=None):
        output1 = self.mhsa_layer(src_h_flatten, pos_embed=pos_embed_h)
        output2 = self.csmhca_layer1(output1, src_m_flatten, pos_embed_h=pos_embed_h, pos_embed_l=pos_embed_m)
        output3 = self.csmhca_layer2(output1, src_l_flatten, pos_embed_h=pos_embed_h, pos_embed_l=pos_embed_l)
        output1 = output1.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        output2 = output2.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        output3 = output3.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        output1 = self.conv(torch.cat([output1, output2, output3], dim=1))

        return output1, output2, output3


class Encoder(nn.Module):   # encoder
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_high_idx=[2],
                 use_encoder_low_idx=[1],
                 num_encoder_high_layers=1,
                 num_encoder_low_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_high_idx = use_encoder_high_idx
        self.use_encoder_low_idx = use_encoder_low_idx
        self.num_encoder_high_layers = num_encoder_high_layers
        self.num_encoder_low_layers = num_encoder_low_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        # channel projection
        self.arfs = arfs_block(dims=[hidden_dim, hidden_dim, hidden_dim], residual=True)
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(nn.Sequential(nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                                                 nn.BatchNorm2d(hidden_dim)
                                                 )
                                   )

        self.mfia = MFIA(hidden_dim=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=enc_act)

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in range(len(self.feat_strides)):
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(self.eval_spatial_size[1] // stride,
                                                                    self.eval_spatial_size[0] // stride,
                                                                    self.hidden_dim,
                                                                    self.pe_temperature
                                                                    )
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, features):    # features: [lo_lv, mid_lv, hi_lv]
        assert len(features) == len(self.in_channels)
        # proj_feats = self.msk(features)
        proj_feats = [self.input_proj[i](feature) for i, feature in enumerate(features)]

        # todo:high_level
        h_h, w_h = proj_feats[2].shape[2:]
        h_m, w_m = proj_feats[1].shape[2:]
        h_l, w_l = proj_feats[0].shape[2:]
        # flatten [B, C, H, W] to [B, HxW, C]
        src_flatten_h = proj_feats[2].flatten(2).permute(0, 2, 1)
        src_flatten_m = proj_feats[1].flatten(2).permute(0, 2, 1)
        src_flatten_l = proj_feats[0].flatten(2).permute(0, 2, 1)
        if self.training or self.eval_spatial_size is None:
            pos_embed_h = self.build_2d_sincos_position_embedding(
                w_h, h_h, self.hidden_dim, self.pe_temperature).to(src_flatten_h.device)
            pos_embed_m = self.build_2d_sincos_position_embedding(
                w_m, h_m, self.hidden_dim, self.pe_temperature).to(src_flatten_m.device)
            pos_embed_l = self.build_2d_sincos_position_embedding(
                w_l, h_l, self.hidden_dim, self.pe_temperature).to(src_flatten_l.device)
        else:
            pos_embed_h = getattr(self, f'pos_embed{2}', None).to(src_flatten_h.device)
            pos_embed_m = getattr(self, f'pos_embed{1}', None).to(src_flatten_m.device)
            pos_embed_l = getattr(self, f'pos_embed{0}', None).to(src_flatten_l.device)

        memorys = self.mfia(src_flatten_h, src_flatten_m, src_flatten_l, h_h, w_h, pos_embed_h, pos_embed_m, pos_embed_l)
        proj_feats[2] = memorys[0]

        # id_level
        proj_feats[1] = proj_feats[1] + F.interpolate(memorys[1], scale_factor=2., mode='bilinear')
        # low_level
        proj_feats[0] = proj_feats[0] + F.interpolate(memorys[2], scale_factor=4., mode='bilinear')

        proj_feats = self.arfs(proj_feats)

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='bilinear')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs


def build_encoder(args):
    # todo: need to set args
    model = Encoder(in_channels=[512, 1024, 2048],
                    feat_strides=[8, 16, 32],
                    hidden_dim=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.,
                    enc_act='gelu',
                    pe_temperature=10000,
                    expansion=1.0,
                    depth_mult=1,
                    act='silu',
                    eval_spatial_size=args.imgsize
                    )
    return model


if __name__ == "__main__":
    device = torch.device("cpu")
    encoder = Encoder().to(device)
    # print(encoder)
    n_parameters = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    features = [torch.randn(2, 512, 80, 80).to(device), torch.randn(2, 1024, 40, 40).to(device), torch.randn(2, 2048, 20, 20).to(device)]
    out = encoder.forward(features)
    for feat in out:
        print(feat.size())
