import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BlockDCT(nn.Module):
    """
    基于卷积的 8x8 块级 DCT 与 IDCT 算子
    """
    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size
        
        # 1. 生成 1D DCT 变换矩阵
        dct_m = torch.zeros(block_size, block_size)
        for k in range(block_size):
            for n in range(block_size):
                if k == 0:
                    dct_m[k, n] = 1.0 / math.sqrt(block_size)
                else:
                    dct_m[k, n] = math.sqrt(2.0 / block_size) * math.cos(math.pi * (2*n + 1) * k / (2*block_size))
        
        # 2. 生成 2D DCT 基准卷积核 (外积)
        # 总共有 block_size * block_size (即64) 个滤波器，每个大小为 8x8
        basis = torch.empty(block_size**2, 1, block_size, block_size)
        idx = 0
        for i in range(block_size):
            for j in range(block_size):
                basis[idx, 0, :, :] = torch.outer(dct_m[i], dct_m[j])
                idx += 1
                
        # 将正交基注册为不可训练的 buffer
        self.register_buffer('dct_weights', basis)

    def apply_block_dct(self, x):
        """
        正向 8x8 块级 DCT
        输入: x [B, C, H, W]
        输出: dct_coefs [B, C, 64, H/8, W/8] (64 个通道对应 64 个频点)
        """
        B, C, H, W = x.shape
        # 将 Batch 和 Channel 合并，变成单通道图像送入卷积
        x_reshaped = x.view(B * C, 1, H, W)
        
        # 使用 stride=8 提取不重叠的 8x8 块，并进行内积计算
        # 输出形状: [B*C, 64, H/8, W/8]
        dct_coefs = F.conv2d(x_reshaped, weight=self.dct_weights, stride=self.block_size)
        
        # 重新 Reshape 回来，方便后续针对每个频点做加权
        return dct_coefs.view(B, C, self.block_size**2, H // self.block_size, W // self.block_size)

    def apply_inverse_block_dct(self, dct_coefs):
        """
        逆向 8x8 块级 IDCT
        输入: dct_coefs [B, C, 64, H/8, W/8]
        输出: x_recon [B, C, H, W]
        """
        B, C, num_freqs, H_8, W_8 = dct_coefs.shape
        # 将 Batch 和 Channel 合并
        coefs_reshaped = dct_coefs.view(B * C, num_freqs, H_8, W_8)
        
        # 由于 DCT 矩阵是严格正交的，逆变换等价于转置！
        # 因此直接使用 conv_transpose2d 配合完全相同的权重即可完美还原
        x_recon = F.conv_transpose2d(coefs_reshaped, weight=self.dct_weights, stride=self.block_size)
        
        return x_recon.view(B, C, x_recon.shape[-2], x_recon.shape[-1])