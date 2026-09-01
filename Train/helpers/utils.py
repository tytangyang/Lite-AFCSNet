import numpy as np
import torch
from torch.distributions.beta import Beta
import random
import torch.distributions as dist

def fre_mixstyle(x, p=0.4, alpha=0.3, eps=1e-6):
    """标准的Fre-MixStyle实现"""
    if np.random.rand() > p:
        return x
    batch_size = x.size(0)

    # frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True)
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()
    f_mu, f_sig = f_mu.detach(), f_sig.detach()
    x_normed = (x - f_mu) / f_sig
    
    lmda = dist.Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)
    perm = torch.randperm(batch_size).to(x.device)
    
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)
    
    x = x_normed * sig_mix + mu_mix
    return x

def randin(x, lambda_val=0.01, phi_val=0.01, eps=1e-6):
    """RandIN实现 - 向统计量添加高斯噪声"""
    batch_size, channels = x.size(0), x.size(1)
    
    # 计算频率方向统计量（与Fre-MixStyle保持一致）
    f_mu = x.mean(dim=[1, 3], keepdim=True)  # (B, 1, H, 1)
    f_var = x.var(dim=[1, 3], keepdim=True)  # (B, 1, H, 1)
    f_sig = (f_var + eps).sqrt()
    
    # 阻断梯度
    f_mu, f_sig = f_mu.detach(), f_sig.detach()
    
    # 添加高斯噪声 - 公式(5)(6)
    R_sigma = f_sig + torch.randn_like(f_sig) * lambda_val
    R_mu = f_mu + torch.randn_like(f_mu) * phi_val
    
    # 应用RandIN - 公式(4)
    x_normed = (x - f_mu) / f_sig
    x = x_normed * R_sigma + R_mu
    
    return x

def probabilistic(x, mixstyle_prob=0.7, p=0.4, alpha=0.3, 
                          lambda_val=0.05, phi_val=0.05):
    """
    概率选择策略
    以mixstyle_prob的概率使用Fre-MixStyle，否则使用RandIN
    
    参数:
        x: 输入特征张量
        mixstyle_prob: 使用MixStyle的概率 (默认0.7)
        p: MixStyle的激活概率
        alpha: MixStyle的Beta分布参数
        lambda_val: RandIN的标准差噪声强度
        phi_val: RandIN的均值噪声强度
    """
    if random.random() < mixstyle_prob:
        # 使用Fre-MixStyle (强制应用，p=1.0确保激活)
        return fre_mixstyle(x, p=1.0, alpha=alpha)
    else:
        # 使用RandIN
        return randin(x, lambda_val=lambda_val, phi_val=phi_val)