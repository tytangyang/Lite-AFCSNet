import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate, stride):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)

        exp_conv = Conv2dNormActivation(
            in_channels,
            exp_channels,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            inplace=False,
        )
        depth_conv = Conv2dNormActivation(
            exp_channels,
            exp_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=exp_channels,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            inplace=False,
        )
        proj_conv = Conv2dNormActivation(
            exp_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,
            inplace=False,
        )
        self.after_block_activation = nn.ReLU()

        if in_channels == out_channels:
            self.use_shortcut = True
            if stride == 1 or stride == (1, 1):
                self.shortcut = nn.Sequential()
            else:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.Sequential(),
                )
        else:
            self.use_shortcut = False

        self.block = nn.Sequential(exp_conv, depth_conv, proj_conv)

    def forward(self, x):
        if self.use_shortcut:
            x = self.block(x) + self.shortcut(x)
        else:
            x = self.block(x)
        return self.after_block_activation(x)


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_classes = config["n_classes"]
        in_channels = config["in_channels"]
        base_channels = config["base_channels"]
        channels_multiplier = config["channels_multiplier"]
        expansion_rate = config["expansion_rate"]
        n_blocks = config["n_blocks"]
        strides = config["strides"]
        n_stages = len(n_blocks)

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [
            make_divisible(base_channels * channels_multiplier ** stage_id, 8)
            for stage_id in range(n_stages)
        ]
        self.total_block_count = 0

        self.in_c = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                channels_per_stage[0] // 4,
                activation_layer=torch.nn.ReLU,
                kernel_size=3,
                stride=2,
                inplace=False,
            ),
            Conv2dNormActivation(
                channels_per_stage[0] // 4,
                channels_per_stage[0],
                activation_layer=torch.nn.ReLU,
                kernel_size=3,
                stride=2,
                inplace=False,
            ),
        )

        self.stages = nn.Sequential()
        for stage_id in range(n_stages):
            stage = self._make_stage(
                channels_per_stage[stage_id],
                channels_per_stage[stage_id + 1],
                n_blocks[stage_id],
                strides=strides,
                expansion_rate=expansion_rate,
            )
            self.stages.add_module(f"s{stage_id + 1}", stage)

        self.feed_forward = nn.Sequential(
            nn.Conv2d(
                channels_per_stage[-1],
                n_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(n_classes),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, strides, expansion_rate):
        stage = nn.Sequential()
        for _ in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f"b{block_id}"
            self.total_block_count += 1
            stride = strides.get(bname, (1, 1))
            stage.add_module(
                bname,
                Block(in_channels, out_channels, expansion_rate, stride),
            )
            in_channels = out_channels
        return stage

    def _forward_conv(self, x):
        x = self.in_c(x)
        return self.stages(x)

    def forward(self, x):
        x = self._forward_conv(x)
        x = self.feed_forward(x)
        return x.squeeze(2).squeeze(2)


def get_model(
    n_classes=10,
    in_channels=1,
    base_channels=32,
    channels_multiplier=1.8,
    expansion_rate=2.1,
    n_blocks=(3, 2, 1),
    strides=None,
):

    if strides is None:
        strides = {
            "b2": (1, 1),
            "b3": (1, 2),
            "b4": (2, 1),
        }
    return Network(
        {
            "n_classes": n_classes,
            "in_channels": in_channels,
            "base_channels": base_channels,
            "channels_multiplier": channels_multiplier,
            "expansion_rate": expansion_rate,
            "n_blocks": n_blocks,
            "strides": strides,
        }
    )
