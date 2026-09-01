# V4: Stage2 large-kernel AsymDW version. Default b4/b5 use 5x1 + 1x3 DWConv.
import torch
import torch.nn as nn
from .fusion import (
    AFF,
    iAFF,
    DAF,
    ConvBNAct,
    PointwiseProjection,
    AsymmetricDepthwiseConv,
    get_valid_groups,
)
from .helpers.utils import make_divisible


SUPPORTED_CONV_TYPES = [
    "standard",      # original: 1x1 expand + 3x3 DWConv + 1x1 project
    "pointwise",     # remove 3x3 DWConv; use pointwise conv only
    "channelwise",   # depthwise/channelwise conv with a cheap grouped adapter
    "group",         # replace depthwise conv by normal KxK group conv
    "group_pw",      # grouped pointwise expand/project
    "shuffle_pw",    # grouped pointwise expand/project + channel shuffle
    "sparse",        # sparsely factorized pointwise expand/project
]


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


def _is_stride_one(stride):
    return stride == 1 or stride == (1, 1)


class ChannelwiseOnlyBlock(nn.Module):
    """
    Very cheap channelwise/depthwise-only block.

    This block removes the inverted expansion and dense pointwise mixing.
    If channel numbers differ, a grouped 1x1 adapter is used only for dimension matching.
    It is mainly provided for ablation; accuracy may drop obviously.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        adapter_groups=4,
    ):
        super(ChannelwiseOnlyBlock, self).__init__()

        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            inplace=False,
        )

        if in_channels != out_channels:
            groups = get_valid_groups(in_channels, out_channels, adapter_groups)
            self.adapter = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=groups,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None,
                inplace=False,
            )
        else:
            self.adapter = nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.adapter(x)
        return x


class BlockWithFusion(nn.Module):
    """
    Inverted bottleneck block with selectable lightweight convolution.

    conv_type:
        standard:
            original block:
            1x1 expand -> 3x3 depthwise -> 1x1 project

        pointwise:
            1x1 expand with stride -> 1x1 project.
            This removes local 3x3 time-frequency modelling.

        channelwise:
            3x3 depthwise/channelwise -> optional grouped 1x1 adapter.
            Very low MACs, but weak channel interaction.

        group:
            1x1 expand -> 3x3 group conv -> 1x1 project.
            Note that small groups may not reduce MACs compared with depthwise.

        group_pw:
            grouped 1x1 expand/project + normal depthwise.

        shuffle_pw:
            grouped 1x1 expand/project + channel shuffle + normal depthwise.
            This is the recommended default for MAC reduction.

        sparse:
            factorized 1x1 expand/project + normal depthwise.

    V1/V3/V4 improvement:
        if use_asym_dw=True, the normal 3x3 depthwise conv is replaced by
        configurable asymmetric depthwise conv with SiLU after the DW part.
        V4 uses stage-wise kernels: normal early blocks use (3,3),
        and late stage2 blocks b4/b5 use (5,3) by default.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion_rate,
        stride,
        fuse_type="DAF",
        conv_type="shuffle_pw",
        pw_groups=4,
        group_conv_groups=4,
        sparse_ratio=0.5,
        use_asym_dw=True,
        dw_activation="silu",
        asym_dw_kernel_size=(3, 3),
        block_name=None,
    ):
        super(BlockWithFusion, self).__init__()

        conv_type = conv_type.lower()
        if conv_type not in SUPPORTED_CONV_TYPES:
            raise ValueError(
                f"Unsupported conv_type={conv_type}. "
                f"Choose from {SUPPORTED_CONV_TYPES}."
            )

        self.conv_type = conv_type
        exp_channels = make_divisible(in_channels * expansion_rate, 8)
        self.use_asym_dw = bool(use_asym_dw)
        self.dw_activation = dw_activation
        self.asym_dw_kernel_size = asym_dw_kernel_size
        self.block_name = block_name

        # ----------------------------------------------------
        # Build main convolution block
        # ----------------------------------------------------
        if conv_type == "channelwise":
            self.block = ChannelwiseOnlyBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                adapter_groups=pw_groups,
            )

        else:
            # Pointwise expand.
            # For pointwise-only mode, stride is moved to the expand layer because there is no depthwise layer.
            expand_stride = stride if conv_type == "pointwise" else 1
            pointwise_mode = conv_type if conv_type in ["group_pw", "shuffle_pw", "sparse"] else "standard"

            exp_conv = PointwiseProjection(
                in_channels,
                exp_channels,
                mode=pointwise_mode,
                stride=expand_stride,
                pw_groups=pw_groups,
                sparse_ratio=sparse_ratio,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU,
                bias=False,
            )

            middle_layers = []

            if conv_type == "pointwise":
                # No 3x3 operation in this ablation.
                pass

            elif conv_type == "group":
                groups = get_valid_groups(exp_channels, exp_channels, group_conv_groups)
                middle_layers.append(
                    ConvBNAct(
                        exp_channels,
                        exp_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        groups=groups,
                        norm_layer=nn.BatchNorm2d,
                        activation_layer=nn.ReLU,
                        inplace=False,
                    )
                )

            else:
                # standard / group_pw / shuffle_pw / sparse all keep local DW modelling.
                # V1/V3/V4 improvement: replace 3x3 DWConv with 3x1 + 1x3 AsymDWConv.
                if self.use_asym_dw:
                    dw_act = nn.SiLU if self.dw_activation.lower() == "silu" else nn.ReLU
                    middle_layers.append(
                        AsymmetricDepthwiseConv(
                            exp_channels,
                            stride=stride,
                            kernel_size=self.asym_dw_kernel_size,
                            norm_layer=nn.BatchNorm2d,
                            activation_layer=dw_act,
                            bias=False,
                            inplace=False,
                        )
                    )
                else:
                    middle_layers.append(
                        ConvBNAct(
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
                    )

            proj_conv = PointwiseProjection(
                exp_channels,
                out_channels,
                mode=pointwise_mode,
                stride=1,
                pw_groups=pw_groups,
                sparse_ratio=sparse_ratio,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None,
                bias=False,
            )

            self.block = nn.Sequential(exp_conv, *middle_layers, proj_conv)

        self.after_block_activation = nn.ReLU()

        # ----------------------------------------------------
        # Fusion module
        # ----------------------------------------------------
        self.fuse_type = fuse_type
        if fuse_type == "AFF":
            self.fuse_mode = AFF(channels=out_channels)
        elif fuse_type == "iAFF":
            self.fuse_mode = iAFF(channels=out_channels)
        elif fuse_type == "DAF":
            self.fuse_mode = DAF()
        else:
            self.fuse_mode = None

        # ----------------------------------------------------
        # Shortcut. Keep the original rule: shortcut only when channels match.
        # ----------------------------------------------------
        if in_channels == out_channels:
            self.use_shortcut = True
            if _is_stride_one(stride):
                self.shortcut = nn.Sequential()
            else:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.Sequential(),
                )
        else:
            self.use_shortcut = False
            self.shortcut = None

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.use_shortcut:
            identity = self.shortcut(identity)
            if self.fuse_mode is not None:
                out = self.fuse_mode(out, identity)
            else:
                out = out + identity

        out = self.after_block_activation(out)
        return out


class TwoStageNetworkWithAFF(nn.Module):
    def __init__(self, config):
        super(TwoStageNetworkWithAFF, self).__init__()

        n_classes = config["n_classes"]
        in_channels = config["in_channels"]
        base_channels = config["base_channels"]
        channels_multiplier = config["channels_multiplier"]
        expansion_rate = config["expansion_rate"]
        n_blocks = config["n_blocks"]
        strides = config["strides"]

        # Only keep the first two stages.
        n_stages = 2

        fuse_type = config.get("fuse_type", "AFF")
        conv_type = config.get("conv_type", "shuffle_pw")
        pw_groups = config.get("pw_groups", 4)
        group_conv_groups = config.get("group_conv_groups", 4)
        sparse_ratio = config.get("sparse_ratio", 0.5)
        use_asym_dw = config.get("use_asym_dw", True)
        dw_activation = config.get("dw_activation", "silu")
        use_maxavg_stage_fusion = config.get("use_maxavg_stage_fusion", True)
        asym_dw_kernel_size = config.get("asym_dw_kernel_size", (3, 3))
        large_asym_dw_kernel_size = config.get("large_asym_dw_kernel_size", (5, 3))
        large_kernel_blocks = tuple(config.get("large_kernel_blocks", ("b4", "b5")))

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [
            make_divisible(base_channels * channels_multiplier ** stage_id, 8)
            for stage_id in range(n_stages)
        ]

        self.total_block_count = 0
        self.channels_per_stage = channels_per_stage
        self.conv_type = conv_type
        self.use_asym_dw = bool(use_asym_dw)
        self.use_maxavg_stage_fusion = bool(use_maxavg_stage_fusion)
        self.asym_dw_kernel_size = asym_dw_kernel_size
        self.large_asym_dw_kernel_size = large_asym_dw_kernel_size
        self.large_kernel_blocks = large_kernel_blocks

        # Learnable Max/Avg stage alignment coefficient.
        # sigmoid(0)=0.5, i.e. equal Max/Avg mixture at initialization.
        self.stage_pool_alpha = nn.Parameter(torch.zeros(1))

        # Stem for Mel-spectrogram input, e.g. [B, 1, 256, 64].
        self.in_c = nn.Sequential(
            ConvBNAct(
                in_channels,
                channels_per_stage[0] // 4,
                activation_layer=torch.nn.ReLU,
                kernel_size=3,
                stride=2,
                padding=1,
                inplace=False,
            ),
            ConvBNAct(
                channels_per_stage[0] // 4,
                channels_per_stage[0],
                activation_layer=torch.nn.ReLU,
                kernel_size=3,
                stride=2,
                padding=1,
                inplace=False,
            ),
        )

        self.stage1 = self._make_stage(
            channels_per_stage[0],
            channels_per_stage[1],
            n_blocks[0],
            strides=strides,
            expansion_rate=expansion_rate,
            fuse_type=fuse_type,
            conv_type=conv_type,
            pw_groups=pw_groups,
            group_conv_groups=group_conv_groups,
            sparse_ratio=sparse_ratio,
            use_asym_dw=use_asym_dw,
            dw_activation=dw_activation,
            asym_dw_kernel_size=asym_dw_kernel_size,
            large_asym_dw_kernel_size=large_asym_dw_kernel_size,
            large_kernel_blocks=large_kernel_blocks,
        )

        self.stage2 = self._make_stage(
            channels_per_stage[1],
            channels_per_stage[2],
            n_blocks[1],
            strides=strides,
            expansion_rate=expansion_rate,
            fuse_type=fuse_type,
            conv_type=conv_type,
            pw_groups=pw_groups,
            group_conv_groups=group_conv_groups,
            sparse_ratio=sparse_ratio,
            use_asym_dw=use_asym_dw,
            dw_activation=dw_activation,
            asym_dw_kernel_size=asym_dw_kernel_size,
            large_asym_dw_kernel_size=large_asym_dw_kernel_size,
            large_kernel_blocks=large_kernel_blocks,
        )

        # Stage-level AFF fusion: stage1 -> stage2.
        self.aff_fusion = AFF(channels=channels_per_stage[2])

        if channels_per_stage[1] != channels_per_stage[2]:
            self.channel_adjust = nn.Conv2d(
                channels_per_stage[1],
                channels_per_stage[2],
                kernel_size=1,
                bias=False,
            )
        else:
            self.channel_adjust = None

        self.detection_head = nn.Sequential(
            nn.Conv2d(
                channels_per_stage[2],
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

    def _make_stage(
        self,
        in_channels,
        out_channels,
        n_blocks,
        strides,
        expansion_rate,
        fuse_type="DAF",
        conv_type="shuffle_pw",
        pw_groups=4,
        group_conv_groups=4,
        sparse_ratio=0.5,
        use_asym_dw=True,
        dw_activation="silu",
        asym_dw_kernel_size=(3, 3),
        large_asym_dw_kernel_size=(5, 3),
        large_kernel_blocks=("b4", "b5"),
    ):
        stage = nn.Sequential()
        for _ in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f"b{block_id}"
            self.total_block_count += 1

            if bname in strides:
                stride = strides[bname]
            else:
                stride = (1, 1)

            current_asym_kernel = (
                large_asym_dw_kernel_size if bname in large_kernel_blocks else asym_dw_kernel_size
            )

            block = self._make_block(
                in_channels,
                out_channels,
                stride=stride,
                expansion_rate=expansion_rate,
                fuse_type=fuse_type,
                conv_type=conv_type,
                pw_groups=pw_groups,
                group_conv_groups=group_conv_groups,
                sparse_ratio=sparse_ratio,
                use_asym_dw=use_asym_dw,
                dw_activation=dw_activation,
                asym_dw_kernel_size=current_asym_kernel,
                block_name=bname,
            )
            stage.add_module(bname, block)
            in_channels = out_channels

        return stage

    def _make_block(
        self,
        in_channels,
        out_channels,
        stride,
        expansion_rate,
        fuse_type="DAF",
        conv_type="shuffle_pw",
        pw_groups=4,
        group_conv_groups=4,
        sparse_ratio=0.5,
        use_asym_dw=True,
        dw_activation="silu",
        asym_dw_kernel_size=(3, 3),
        block_name=None,
    ):
        return BlockWithFusion(
            in_channels,
            out_channels,
            expansion_rate,
            stride,
            fuse_type=fuse_type,
            conv_type=conv_type,
            pw_groups=pw_groups,
            group_conv_groups=group_conv_groups,
            sparse_ratio=sparse_ratio,
            use_asym_dw=use_asym_dw,
            dw_activation=dw_activation,
            asym_dw_kernel_size=asym_dw_kernel_size,
            block_name=block_name,
        )

    def forward(self, x):
        x = self.in_c(x)

        stage1_out = self.stage1(x)
        stage2_out = self.stage2(stage1_out)

        # Align stage1 output to stage2 output before AFF fusion.
        # V1/V3/V4 improvement: use a learnable Max/Avg mixture instead of Avg-only alignment.
        if stage1_out.size() != stage2_out.size():
            target_size = stage2_out.size()[2:]
            if self.use_maxavg_stage_fusion:
                stage1_avg = nn.functional.adaptive_avg_pool2d(stage1_out, target_size)
                stage1_max = nn.functional.adaptive_max_pool2d(stage1_out, target_size)
                alpha = torch.sigmoid(self.stage_pool_alpha)
                stage1_out_adapted = alpha * stage1_avg + (1.0 - alpha) * stage1_max
            else:
                stage1_out_adapted = nn.functional.adaptive_avg_pool2d(
                    stage1_out,
                    target_size,
                )

            if self.channel_adjust is not None:
                stage1_out_adapted = self.channel_adjust(stage1_out_adapted)
            fused_features = self.aff_fusion(stage2_out, stage1_out_adapted)
        else:
            fused_features = self.aff_fusion(stage2_out, stage1_out)

        output = self.detection_head(fused_features)
        logits = output.squeeze(2).squeeze(2)
        return logits


def get_two_stage_model(
    n_classes=10,
    in_channels=1,
    base_channels=32,
    channels_multiplier=2.3,
    expansion_rate=3.0,
    n_blocks=(3, 2),
    strides=None,
    fuse_type="AFF",
    conv_type="shuffle_pw",
    pw_groups=4,
    group_conv_groups=4,
    sparse_ratio=0.5,
    use_asym_dw=True,
    dw_activation="silu",
    use_maxavg_stage_fusion=True,
    asym_dw_kernel_size=(3, 3),
    large_asym_dw_kernel_size=(5, 3),
    large_kernel_blocks=("b4", "b5"),
):
    """
    Two-stage network + AFF/LC-AFF fusion + selectable lightweight conv.

    Parameters
    ----------
    conv_type:
        "standard"    : original block, used as baseline.
        "pointwise"   : remove depthwise 3x3, only pointwise operations.
        "channelwise" : depthwise/channelwise-only with cheap grouped adapter.
        "group"       : KxK group convolution in the middle.
        "group_pw"    : grouped pointwise expand/project.
        "shuffle_pw"  : grouped pointwise expand/project + channel shuffle.
        "sparse"      : factorized pointwise expand/project.

    V4 defaults:
        conv_type="shuffle_pw"
        use_asym_dw=True
        dw_activation="silu"
        use_maxavg_stage_fusion=True
        asym_dw_kernel_size=(3, 3)
        large_asym_dw_kernel_size=(5, 3)
        large_kernel_blocks=("b4", "b5")

    Recommended ablation order:
        1. baseline: use_asym_dw=False, use_maxavg_stage_fusion=False
        2. V1-A:    use_asym_dw=True,  use_maxavg_stage_fusion=False
        3. V1-B:    use_asym_dw=True,  use_maxavg_stage_fusion=True
        4. V3-C:    large_kernel_blocks=("b5",), large_asym_dw_kernel_size=(5,3)
        5. V4:      large_kernel_blocks=("b4", "b5"), large_asym_dw_kernel_size=(5,3)
    """

    if strides is None:
        strides = dict(
            b2=(1, 1),
            b3=(1, 2),
            b4=(2, 1),
        )

    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "expansion_rate": expansion_rate,
        "n_blocks": n_blocks,
        "strides": strides,
        "fuse_type": fuse_type,
        "conv_type": conv_type,
        "pw_groups": pw_groups,
        "group_conv_groups": group_conv_groups,
        "sparse_ratio": sparse_ratio,
        "use_asym_dw": use_asym_dw,
        "dw_activation": dw_activation,
        "use_maxavg_stage_fusion": use_maxavg_stage_fusion,
        "asym_dw_kernel_size": asym_dw_kernel_size,
        "large_asym_dw_kernel_size": large_asym_dw_kernel_size,
        "large_kernel_blocks": large_kernel_blocks,
    }

    model = TwoStageNetworkWithAFF(model_config)
    return model
