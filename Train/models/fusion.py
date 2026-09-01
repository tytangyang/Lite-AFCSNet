import math
import torch
import torch.nn as nn


# ============================================================
# Basic lightweight convolution utilities
# ============================================================


def get_valid_groups(in_channels: int, out_channels: int, groups: int) -> int:
    groups = int(groups)
    if groups <= 1:
        return 1
    g = math.gcd(int(in_channels), int(out_channels))
    g = math.gcd(g, groups)
    return max(1, g)


def _build_activation(activation_layer, inplace: bool = False):
    if activation_layer is None:
        return None
    try:
        return activation_layer(inplace=inplace)
    except TypeError:
        return activation_layer()


class ConvBNAct(nn.Sequential):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=1,
        stride=1,
        padding=0,
        groups: int = 1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        bias: bool = False,
        inplace: bool = False,
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        act = _build_activation(activation_layer, inplace=inplace)
        if act is not None:
            layers.append(act)
        super().__init__(*layers)


class ChannelShuffle(nn.Module):


    def __init__(self, groups: int = 4):
        super(ChannelShuffle, self).__init__()
        self.groups = int(groups)

    def forward(self, x):
        if self.groups <= 1:
            return x

        b, c, h, w = x.size()
        if c % self.groups != 0:
            return x

        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x


class AsymmetricDepthwiseConv(nn.Module):


    def __init__(
        self,
        channels: int,
        stride=1,
        kernel_size=(3, 3),
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.SiLU,
        bias: bool = False,
        inplace: bool = False,
    ):
        super(AsymmetricDepthwiseConv, self).__init__()

        if isinstance(stride, int):
            stride_f, stride_t = stride, stride
        else:
            stride_f, stride_t = stride

        if isinstance(kernel_size, int):
            k_f, k_t = kernel_size, kernel_size
        else:
            k_f, k_t = kernel_size

        if k_f % 2 == 0 or k_t % 2 == 0:
            raise ValueError("AsymmetricDepthwiseConv expects odd kernel sizes, e.g. (3,3), (5,3), or (5,5).")

        self.kernel_size = (int(k_f), int(k_t))

        # Frequency-direction modelling: k_f x 1.
        self.dw_freq = nn.Conv2d(
            channels,
            channels,
            kernel_size=(k_f, 1),
            stride=(stride_f, 1),
            padding=(k_f // 2, 0),
            groups=channels,
            bias=bias,
        )

        # Time-direction modelling: 1 x k_t.
        self.dw_time = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, k_t),
            stride=(1, stride_t),
            padding=(0, k_t // 2),
            groups=channels,
            bias=bias,
        )

        self.norm = norm_layer(channels) if norm_layer is not None else nn.Identity()
        act = _build_activation(activation_layer, inplace=inplace)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x):
        x = self.dw_freq(x)
        x = self.dw_time(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SparseFactorizedPointwiseConv(nn.Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride=1,
        sparse_ratio: float = 0.5,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        final_activation_layer=None,
        bias: bool = False,
    ):
        super(SparseFactorizedPointwiseConv, self).__init__()

        mid_channels = max(1, int(min(in_channels, out_channels) * sparse_ratio))

        self.reduce = ConvBNAct(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            bias=bias,
            inplace=False,
        )
        self.expand = ConvBNAct(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_layer=norm_layer,
            activation_layer=final_activation_layer,
            bias=bias,
            inplace=False,
        )

    def forward(self, x):
        x = self.reduce(x)
        x = self.expand(x)
        return x


class PointwiseProjection(nn.Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "standard",
        stride=1,
        pw_groups: int = 4,
        sparse_ratio: float = 0.5,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        bias: bool = False,
    ):
        super(PointwiseProjection, self).__init__()
        mode = mode.lower()
        self.mode = mode

        if mode in ["standard", "pointwise", "channelwise", "group"]:
            self.proj = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                bias=bias,
                inplace=False,
            )

        elif mode in ["group_pw", "shuffle_pw"]:
            groups = get_valid_groups(in_channels, out_channels, pw_groups)
            layers = [
                ConvBNAct(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    bias=bias,
                    inplace=False,
                )
            ]
            if mode == "shuffle_pw":
                layers.append(ChannelShuffle(groups))
            self.proj = nn.Sequential(*layers)
            self.groups = groups

        elif mode == "sparse":
            self.proj = SparseFactorizedPointwiseConv(
                in_channels,
                out_channels,
                stride=stride,
                sparse_ratio=sparse_ratio,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU,
                final_activation_layer=activation_layer,
                bias=bias,
            )

        else:
            raise ValueError(
                f"Unsupported pointwise mode: {mode}. "
                "Choose from ['standard', 'group_pw', 'shuffle_pw', 'sparse']."
            )

    def forward(self, x):
        return self.proj(x)


# ============================================================
# AFF family
# ============================================================


class DAF(nn.Module):


    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):


    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = max(int(channels // r), 1)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual

        xl = self.local_att(xa)
        xg = self.global_att(xa)
        wei = self.sigmoid(xl + xg)
        xi = x * wei + residual * (1.0 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        wei2 = self.sigmoid(xl2 + xg2)
        xo = x * wei2 + residual * (1.0 - wei2)
        return xo


class OriginalAFF(nn.Module):


    def __init__(self, channels=64, r=4):
        super(OriginalAFF, self).__init__()
        inter_channels = max(int(channels // r), 1)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        wei = self.sigmoid(xl + xg)
        xo = 2.0 * x * wei + 2.0 * residual * (1.0 - wei)
        return xo


class ECAChannelAttention(nn.Module):


    def __init__(self, channels=64, kernel_size=3):
        super(ECAChannelAttention, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd, e.g. 3, 5, or 7.")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                     # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)      # [B, 1, C]
        y = self.conv(y)                         # [B, 1, C]
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)    # [B, C, 1, 1]
        return y


class LCAFF(nn.Module):


    def __init__(self, channels=64, r=4, kernel_size=3, scale=2.0):
        super(LCAFF, self).__init__()
        # r is kept only for API compatibility with AFF(channels=..., r=...).
        self.channels = channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.channel_att = ECAChannelAttention(channels=channels, kernel_size=kernel_size)

    def forward(self, x, residual):
        if x.shape != residual.shape:
            raise ValueError(
                "LCAFF requires x and residual to have the same shape. "
                f"Got x={tuple(x.shape)}, residual={tuple(residual.shape)}. "
                "Please align spatial size and channels before fusion."
            )

        xa = x + residual
        wei = self.channel_att(xa)
        xo = self.scale * x * wei + self.scale * residual * (1.0 - wei)
        return xo


class AFF(LCAFF):


    def __init__(self, channels=64, r=4, kernel_size=3, scale=2.0):
        super(AFF, self).__init__(
            channels=channels,
            r=r,
            kernel_size=kernel_size,
            scale=scale,
        )


class MS_CAM(nn.Module):


    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = max(int(channels // r), 1)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        wei = self.sigmoid(xl + xg)
        return x * wei
