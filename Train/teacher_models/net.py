import torch
import torch.nn as nn
from .fusion import AFF, iAFF, DAF
from torchvision.ops.misc import Conv2dNormActivation

from .helpers.utils import make_divisible


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


class BlockWithFusion(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            expansion_rate,
            stride,
            fuse_type='DAF'
    ):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)

        # 原有的三个卷积层保持不变
        exp_conv = Conv2dNormActivation(in_channels,
                                        exp_channels,
                                        kernel_size=1,
                                        stride=1,
                                        norm_layer=nn.BatchNorm2d,
                                        activation_layer=nn.ReLU,
                                        inplace=False
                                        )

        depth_conv = Conv2dNormActivation(exp_channels,
                                          exp_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=1,
                                          groups=exp_channels,
                                          norm_layer=nn.BatchNorm2d,
                                          activation_layer=nn.ReLU,
                                          inplace=False
                                          )

        proj_conv = Conv2dNormActivation(exp_channels,
                                         out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         norm_layer=nn.BatchNorm2d,
                                         activation_layer=None,
                                         inplace=False
                                         )
        
        self.after_block_activation = nn.ReLU()

        # 添加融合模块
        self.fuse_type = fuse_type
        if fuse_type == 'AFF':
            self.fuse_mode = AFF(channels=out_channels)
        elif fuse_type == 'iAFF':
            self.fuse_mode = iAFF(channels=out_channels)
        elif fuse_type == 'DAF':
            self.fuse_mode = DAF()
        else:
            self.fuse_mode = None

        # 快捷连接逻辑保持不变
        if in_channels == out_channels:
            self.use_shortcut = True
            if stride == 1 or stride == (1, 1):
                self.shortcut = nn.Sequential()
            else:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.Sequential()
                )
        else:
            self.use_shortcut = False

        self.block = nn.Sequential(
            exp_conv,
            depth_conv,
            proj_conv
        )

    def forward(self, x):
        identity = x
        
        out = self.block(x)
        
        if self.use_shortcut:
            if self.fuse_mode is not None:
                identity = self.shortcut(identity)
                out = self.fuse_mode(out, identity)
            else:
                out = out + self.shortcut(identity)
        else:
            pass
            
        out = self.after_block_activation(out)
        return out


class TwoStageNetworkWithAFF(nn.Module):
    def __init__(self, config):
        super(TwoStageNetworkWithAFF, self).__init__()
        # 配置参数
        n_classes = config['n_classes']
        in_channels = config['in_channels']
        base_channels = config['base_channels']
        channels_multiplier = config['channels_multiplier']
        expansion_rate = config['expansion_rate']
        n_blocks = config['n_blocks']
        strides = config['strides']
        
        # 只使用前两个阶段
        n_stages = 2
        
        # AFF融合配置
        fuse_type = config.get('fuse_type', 'AFF')  # 默认使用AFF

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [make_divisible(base_channels * channels_multiplier ** stage_id, 8)
                                                for stage_id in range(n_stages)]
        self.total_block_count = 0

        # 初始卷积层 - 针对(1,256,64)输入优化
        self.in_c = nn.Sequential(
            Conv2dNormActivation(in_channels,
                                 channels_per_stage[0] // 4,
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1,  # 明确指定padding
                                 inplace=False
                                 ),
            Conv2dNormActivation(channels_per_stage[0] // 4,
                                 channels_per_stage[0],
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1,  # 明确指定padding
                                 inplace=False
                                 ),
        )

        # 只创建阶段1和阶段2
        self.stage1 = self._make_stage(channels_per_stage[0],
                                     channels_per_stage[1],
                                     n_blocks[0],
                                     strides=strides,
                                     expansion_rate=expansion_rate,
                                     fuse_type=fuse_type
                                     )
        
        self.stage2 = self._make_stage(channels_per_stage[1],
                                     channels_per_stage[2],
                                     n_blocks[1],
                                     strides=strides,
                                     expansion_rate=expansion_rate,
                                     fuse_type=fuse_type
                                     )

        # 添加AFF融合模块，融合stage1和stage2的特征
        self.aff_fusion = AFF(channels=channels_per_stage[2])
        
        # 预先定义通道调整卷积层，避免在forward中动态创建
        if channels_per_stage[1] != channels_per_stage[2]:
            self.channel_adjust = nn.Conv2d(
                channels_per_stage[1], 
                channels_per_stage[2], 
                kernel_size=1
            )
        else:
            self.channel_adjust = None

        # 探测头（分类头）
        ff_list = [
            nn.Conv2d(
                channels_per_stage[2],
                n_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(n_classes),
            nn.AdaptiveAvgPool2d((1, 1))
        ]

        self.detection_head = nn.Sequential(*ff_list)

        self.apply(initialize_weights)

    def _make_stage(self,
                    in_channels,
                    out_channels,
                    n_blocks,
                    strides,
                    expansion_rate,
                    fuse_type='DAF'):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f'b{block_id}'
            self.total_block_count = self.total_block_count + 1
            if bname in strides:
                stride = strides[bname]
            else:
                stride = (1, 1)

            block = self._make_block(
                in_channels,
                out_channels,
                stride=stride,
                expansion_rate=expansion_rate,
                fuse_type=fuse_type
            )
            stage.add_module(bname, block)

            in_channels = out_channels
        return stage

    def _make_block(self,
                    in_channels,
                    out_channels,
                    stride,
                    expansion_rate,
                    fuse_type='DAF'
                    ):

        block = BlockWithFusion(in_channels,
                      out_channels,
                      expansion_rate,
                      stride,
                      fuse_type
                      )
        return block

    def forward(self, x):
        # 初始卷积
        x = self.in_c(x)
        
        # 阶段1
        stage1_out = self.stage1(x)
        
        # 阶段2
        stage2_out = self.stage2(stage1_out)
        
        # AFF融合阶段1和阶段2的特征
        # 注意：这里需要将stage1_out调整到与stage2_out相同的尺寸
        if stage1_out.size() != stage2_out.size():
            # 使用自适应池化调整空间尺寸
            stage1_out_adapted = nn.functional.adaptive_avg_pool2d(stage1_out, stage2_out.size()[2:])
            
            # 如果通道数不同，使用预定义的1x1卷积调整通道数
            if self.channel_adjust is not None:
                stage1_out_adapted = self.channel_adjust(stage1_out_adapted)
                
            fused_features = self.aff_fusion(stage2_out, stage1_out_adapted)
        else:
            fused_features = self.aff_fusion(stage2_out, stage1_out)
        
        # 探测头
        output = self.detection_head(fused_features)
        logits = output.squeeze(2).squeeze(2)
        return logits


def get_two_stage_model(n_classes=10, in_channels=1, base_channels=32, 
                       channels_multiplier=2.3, expansion_rate=3.0,
                       n_blocks=(3, 2), strides=None, fuse_type='AFF'):
    """
    两阶段网络 + AFF融合 + 探测头
    专门针对梅尔频谱图输入(1,256,64)优化
    
    @param n_classes: 类别数
    @param in_channels: 输入通道，梅尔频谱图为1
    @param base_channels: 基础通道数  
    @param channels_multiplier: 通道倍增系数
    @param expansion_rate: 扩展率
    @param n_blocks: 两个阶段的块数，例如(3, 2)
    @param strides: 步长配置
    @param fuse_type: 块内融合类型，但阶段间固定使用AFF
    @return: 两阶段AFF融合网络模型
    """
    
    if strides is None:
        strides = dict(
            b2=(1, 1),
            b3=(1, 2),  # 时间维度下采样
            b4=(2, 1)   # 频率维度下采样
        )

    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "expansion_rate": expansion_rate,
        "n_blocks": n_blocks,
        "strides": strides,
        "fuse_type": fuse_type
    }

    m = TwoStageNetworkWithAFF(model_config)
    return m