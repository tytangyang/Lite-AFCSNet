import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

class SpecAugment(nn.Module):
    """Advanced SpecAugment with more configuration options"""
    def __init__(self, freq_mask_param, time_mask_param, 
                 num_freq_masks=2, num_time_masks=2,
                 freq_mask_max=0, time_mask_max=0):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        
        # 创建多个独立的掩蔽实例
        self.freq_masks = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param, iid_masks=True) 
            for _ in range(num_freq_masks)
        ])
        self.time_masks = nn.ModuleList([
            T.TimeMasking(time_mask_param, iid_masks=True) 
            for _ in range(num_time_masks)
        ])

    def forward(self, x):
        # 随机选择要应用的频率掩蔽数量
        num_freq_to_apply = torch.randint(0, self.num_freq_masks + 1, (1,)).item()
        # 随机选择要应用的时域掩蔽数量
        num_time_to_apply = torch.randint(0, self.num_time_masks + 1, (1,)).item()
        
        # 打乱掩蔽顺序
        freq_indices = torch.randperm(self.num_freq_masks)[:num_freq_to_apply]
        time_indices = torch.randperm(self.num_time_masks)[:num_time_to_apply]
        
        # 应用频率掩蔽
        for idx in freq_indices:
            x = self.freq_masks[idx](x)
        
        # 应用时域掩蔽
        for idx in time_indices:
            x = self.time_masks[idx](x)
            
        return x